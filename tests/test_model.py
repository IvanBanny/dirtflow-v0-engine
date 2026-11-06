import pytest
import torch
from torch.nn import functional as F

from dirtflow_engine.model import coordinates, dense, features, heads, layers, sparse
from dirtflow_engine.model.sparse_flow import BlockStateFlowModel, to_native
from dirtflow_engine.sampling import clean_estimate, support


def test_sparse_convolution_matches_dense():
    torch.manual_seed(8)
    mask = torch.rand(2, 4, 4, 4) > 0.5
    values = torch.randn(2, 4, 4, 4, 3)
    grid = sparse.from_dense_mask(mask, values)
    conv = sparse.SubMConv3d(3, 2)
    out = conv(grid)
    weights = conv.weight.reshape(3, 3, 3, 3, 2).permute(4, 3, 0, 1, 2)
    expected = F.conv3d(
        (values * mask[..., None]).movedim(-1, 1), weights, conv.bias, padding=1
    ).movedim(1, -1)[mask]
    torch.testing.assert_close(out.feats, expected, atol=5e-7, rtol=1e-5)
    index = grid.cache["index"]
    conv(grid.replace_feats(grid.feats * 2))
    assert grid.cache["index"] is index


def test_pool_is_active_child_mean_and_reuses_geometry():
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 3, 3, 3]])
    grid = sparse.SparseGrid(coords, torch.tensor([[2.0], [6.0], [9.0]]), 1, 4)
    coarse, parent = sparse.downsample(grid)
    torch.testing.assert_close(coarse.feats, torch.tensor([[4.0], [9.0]]))
    torch.testing.assert_close(
        sparse.upsample(coarse, parent, grid).feats, torch.tensor([[4.0], [4.0], [9.0]])
    )
    again, _ = sparse.downsample(grid.replace_feats(grid.feats + 1))
    assert again.coords is coarse.coords and again.cache is coarse.cache
    torch.testing.assert_close(again.feats, coarse.feats + 1)


def test_ragged_attention_matches_each_sample():
    torch.manual_seed(3)
    q, k, v = [torch.randn(7, 2, 4) for _ in range(3)]
    lengths = torch.tensor([2, 0, 5])
    result = sparse.varlen_self_attention(q, k, v, lengths)
    expected = []
    for a, b, c in zip(q.split([2, 0, 5]), k.split([2, 0, 5]), v.split([2, 0, 5])):
        if len(a):
            expected.append(
                F.scaled_dot_product_attention(
                    a.transpose(0, 1), b.transpose(0, 1), c.transpose(0, 1)
                ).transpose(0, 1)
            )
    torch.testing.assert_close(result, torch.cat(expected))


def test_native_coordinates_preserve_handedness_and_order():
    grid = sparse.SparseGrid(
        torch.tensor([[0, 0, 1, 3], [0, 2, 3, 0]]), torch.tensor([[5.0], [8.0]]), 1, 4
    )
    native, restore = to_native(grid, "trellis_z_up")
    assert native.coords.tolist() == [[0, 0, 0, 1], [0, 2, 3, 3]]
    torch.testing.assert_close(native.feats[restore], grid.feats)
    value = torch.arange(64).reshape(4, 4, 4)
    torch.testing.assert_close(
        coordinates.to_world_grid(
            coordinates.to_structure_grid(value, "trellis_z_up"), "trellis_z_up"
        ),
        value,
    )


def test_dense_and_sparse_feature_layout():
    before = torch.zeros(1, 3, 3, 3, dtype=torch.long)
    before[0, 1, 1, 1] = 2
    observed = torch.ones_like(before, dtype=torch.bool)
    edit = torch.zeros_like(observed)
    edit[0, 1, 1, 1] = True
    table = torch.arange(24).reshape(3, 8).float()
    x = features.dense_observation_features(before, edit, observed, table)
    assert x.shape == (1, 20, 3, 3, 3)
    assert x[0, :9, 1, 1, 1].count_nonzero() == 0
    torch.testing.assert_close(x[0, 12:, 1, 1, 1], table[2])
    dropped = features.dense_observation_features(
        before, edit, observed, table, context_keep=torch.zeros(1)
    )
    assert dropped.count_nonzero() == 1 and dropped[0, 9, 1, 1, 1] == 1
    neighbors = features.neighbor_observations(torch.tensor([[0, 0, 0, 0]]), observed)
    assert neighbors.tolist() == [[0, 1, 0, 1, 0, 1]]
    out = features.sparse_observation_features(
        torch.ones(1, 8),
        table[:1],
        torch.tensor([False]),
        torch.tensor([[2.0]]),
        torch.ones(1, 1),
        torch.full((1, 8), 3.0),
        table[2:],
        torch.tensor([True]),
        neighbors,
    )
    assert out.shape == (1, 42)
    torch.testing.assert_close(out[0, 17:19], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(out[0, 27:35], table[2])


def test_support_keeps_air_witnesses_and_material_occupancy():
    before = torch.zeros(6, 6, 6, dtype=torch.long)
    edit = torch.zeros_like(before, dtype=torch.bool)
    edit[2:4, 2:4, 2:4] = True
    before[2, 2, 2] = 3
    seen = torch.ones_like(edit)
    logits = torch.full(before.shape, -1.0)
    generated = support(logits, edit, seen, before, "replacement")
    assert torch.equal(generated, ~edit)
    for task in ("repair", "material_only"):
        selected = support(logits, edit, seen, before, task)
        assert selected.sum() == (~edit).sum() + 1
        assert selected[2, 2, 2]


def test_flow_clean_estimate():
    clean, noise = torch.tensor([[1.0, 2.0]]), torch.tensor([[-3.0, 4.0]])
    t = 0.3
    x = (1 - t) * clean + (1e-5 + (1 - 1e-5) * t) * noise
    velocity = (1 - 1e-5) * noise - clean
    torch.testing.assert_close(clean_estimate(x, velocity, t), clean)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_norm(dtype):
    torch.manual_seed(9)
    norm = layers.ChunkedLayerNorm32(7)
    values = torch.randn(31, 7).to(dtype)
    result = norm(values)
    expected = F.layer_norm(values.float(), (7,), norm.weight, norm.bias, norm.eps).to(
        dtype
    )
    torch.testing.assert_close(
        result,
        expected,
        atol=1e-6 if dtype == torch.float32 else 0.016,
        rtol=1e-5 if dtype == torch.float32 else 0.008,
    )


def test_model_shapes_and_checkpoint_keys():
    torch.manual_seed(4)
    flow = dense.StructureFlowModel(
        resolution=2,
        in_channels=32,
        model_channels=24,
        cond_channels=12,
        num_blocks=1,
        num_heads=4,
        self_conditioning=True,
    )
    output = flow(
        torch.randn(1, 32, 2, 2, 2), torch.tensor([500.0]), torch.randn(1, 3, 12)
    )
    assert output.shape == (1, 8, 2, 2, 2) and torch.isfinite(output).all()
    assert "blocks.0.self_attn.to_qkv.weight" in flow.state_dict()
    sparse_flow = BlockStateFlowModel(
        resolution=4,
        in_channels=42,
        model_channels=24,
        cond_channels=12,
        num_blocks=1,
        num_heads=4,
        io_block_channels=(8,),
        num_io_res_blocks=2,
        coordinate_system="trellis_z_up",
    )
    grid = sparse.from_dense_mask(
        torch.ones(1, 4, 4, 4, dtype=torch.bool), torch.randn(1, 4, 4, 4, 42)
    )
    output, io = sparse_flow(
        grid, torch.tensor([500.0]), torch.randn(1, 3, 12), return_io_feats=True
    )
    assert output.feats.shape == (64, 8) and io.shape == (64, 8)
    assert torch.isfinite(output.feats).all()


def test_readout_matches_full_logits_and_excludes_air():
    torch.manual_seed(7)
    head = heads.IndependentCategoricalHead(
        hidden=16, feat_dim=4, output_to_global=torch.tensor([0, 2, 7])
    )
    table, latent, io = torch.randn(8, 8), torch.randn(17, 8), torch.randn(17, 4)
    expected = head.global_ids(head.full_vocab_logits(latent, table, io).argmax(1))
    actual = heads.argmax_over_vocab(head, latent, table, io, chunk=3)
    torch.testing.assert_close(actual, expected)
    assert not (
        heads.argmax_over_vocab(
            head, latent, table, io, forbidden_global_ids=torch.tensor([0])
        )
        == 0
    ).any()
    assert head.target_classes(torch.tensor([0, 1, 2, 7, 8])).tolist() == [
        0,
        -1,
        1,
        2,
        -1,
    ]


def test_clean_estimate_uses_tensor_precision():
    torch.manual_seed(17)
    x, velocity = torch.randn(100, 8), torch.randn(100, 8)
    t = float(torch.linspace(1, 0, 31)[7])
    time = torch.full((100, 1), t)
    expected = (1 - 1e-5) * x - (1e-5 + (1 - 1e-5) * time) * velocity
    assert torch.equal(clean_estimate(x, velocity, t), expected)


def test_support_context_cap_and_tie_order():
    before = torch.zeros(32, 32, 32, dtype=torch.long)
    edit = torch.zeros_like(before, dtype=torch.bool)
    edit[15:17, 15:17, 15:17] = True
    selected = support(
        torch.zeros_like(before).float(),
        edit,
        torch.ones_like(edit),
        before,
        "replacement",
    )
    assert selected.sum() == 8192 + 8
    assert selected[edit].all()
    coords = torch.stack(torch.meshgrid(*[torch.arange(32)] * 3, indexing="ij"), dim=-1)
    distance = torch.maximum(15 - coords, coords - 16).clamp_min(0).amax(-1)
    order = distance.flatten().argsort(stable=True)
    expected = torch.zeros_like(edit).flatten()
    expected[order[:8200]] = True
    assert torch.equal(selected.flatten(), expected)
