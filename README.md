<h1><img src="images/dirtflow.svg" width="48" height="48" alt="" align="absmiddle"> Dirtflow engine</h1>

[Research](https://github.com/IvanBanny/dirtflow-v0) |
[Model](https://huggingface.co/IvanBanny/dirtflow-v0) |
[Engine](https://github.com/IvanBanny/dirtflow-v0-engine) |
[Mod](https://github.com/IvanBanny/dirtflow-v0-mod) |
[Demo](https://huggingface.co/spaces/IvanBanny/dirtflow-v0)

Dirtflow generates Minecraft block edits from a prompt and a snapshot of the
surrounding world. This is the PyTorch inference server for the Dirtflow mod.
It loads the model once and runs one generation at a time.

## Run

Use [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12
and an NVIDIA GPU with BF16 support. The lockfile selects
PyTorch 2.12 with CUDA 13.2. Create the environment outside the checkout:

```bash
git clone https://github.com/IvanBanny/dirtflow-v0-engine.git
cd dirtflow-v0-engine
uv venv --python 3.12 ~/.venvs/dirtflow-engine
UV_PROJECT_ENVIRONMENT=~/.venvs/dirtflow-engine uv sync --frozen
source ~/.venvs/dirtflow-engine/bin/activate
dirtflow-engine
```

The first run downloads the 9.3 GB model from
[IvanBanny/dirtflow-v0](https://huggingface.co/IvanBanny/dirtflow-v0).
[download.py](src/dirtflow_engine/download.py) pins the model revision. Files
live in `$HF_HOME/dirtflow-engine/<revision>`; `HF_HOME` defaults to
`~/.cache/huggingface`. Set it before starting to use another drive.
Later runs use the complete cache without network access. The engine verifies
every file's SHA-256 hash before loading.

To download and verify the model without starting the server or using a GPU:

```bash
dirtflow-engine --download-only
```

An interrupted download resumes when you run the same command again.
`HF_HUB_OFFLINE=1` requires an existing complete cache. To use a bundle you
downloaded separately, run `dirtflow-engine --model /path/to/bundle`.
The directory must contain `bundle.json`, both weight payloads, the block-state
schema and CLIP assets. Local bundles make no network requests.

The server listens on `127.0.0.1:8000`. In the mod, select a region and run
`/dirtflow a stone tower`. Defaults are 30 steps per stage, CFG 5, and one
candidate. The prompt goes directly to CLIP, whose context is 77 tokens.
The [mod README](https://github.com/IvanBanny/dirtflow-v0-mod#install) covers
Minecraft installation, selection and preview controls.

Stages move between CPU and GPU to reduce VRAM use. `--resident` keeps both on
the GPU. `--sequential-guidance` evaluates the two CFG branches separately.
`--no-physics` disables the final placement rules. `--device cpu` runs the same
model on CPU in FP32. `--help` lists the server options.

For a shared server, set `DIRTFLOW_TOKEN` and use `--host 0.0.0.0`. Job routes
then require `Authorization: Bearer <token>`. Put HTTPS in front of the server
when connecting over an untrusted network.

## API

The HTTP envelope is version 2; URLs remain under `/v1`. The binary tensor
format is version 1. The server supports Minecraft 1.21.11.

| Route | Result |
| --- | --- |
| `GET /v1/health` | Readiness and protocol version |
| `POST /v1/generate` | Job ID and queue position |
| `GET /v1/jobs/{id}` | Current phase |
| `GET /v1/jobs/{id}/events` | SSE status, progress, preview, result, error and end events |
| `DELETE /v1/jobs/{id}` | Cooperative cancellation |

Request fields and bounds are in [request.py](src/dirtflow_engine/request.py).
FastAPI also serves the schema at `/docs`. The selection must fit inside a
64-cube with the thickest fitting context shell of 3, 5 or 8 blocks. Each
editable axis is at most 58 blocks. Snapshots must cover that whole rectangle.
Sparse result entries include explicit air for carving; omitted entries stay
unchanged. [voxel.py](src/dirtflow_engine/voxel.py) defines the byte layout.

The queue holds eight waiting requests and returns HTTP 429 when full.
Completed jobs remain available for up to 15 minutes, with at most 32 retained.
SSE clients can resume with `Last-Event-ID`. Progress and previews retain only
their latest values; terminal results remain available until the job expires.
Cancellation holds the worker until the sampler thread exits.

## Development

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/dirtflow-engine uv sync --frozen --extra dev
source ~/.venvs/dirtflow-engine/bin/activate
pytest -m 'not integration'
ruff check --no-cache src tests
ruff format --check --no-cache src tests
mypy --cache-dir /tmp/dirtflow-mypy src
```

`server.py` and `jobs.py` own HTTP and job state. `engine.py` places snapshots,
runs candidates and encodes the result. `sampling.py` owns the two Euler loops;
`model/` contains the checkpoint-compatible layers and loader. `schema.py`
normalizes game states. `physics.py` applies a small set of placement rules;
it does not simulate Minecraft ticks.

Optional checks use a local bundle or the mod's compiled runtime classpath:

```bash
DIRTFLOW_TEST_BUNDLE=/path/to/bundle pytest tests/test_inference.py
DIRTFLOW_MOD_CLASSPATH=/path/to/mod/classes:/path/to/dependencies pytest tests/test_java.py
```

CUDA results can depend on hardware and PyTorch kernels. CPU tests check the
protocol and numerical components; model tests exercise actual bundle loading
and generation. The game client's rendering and world application need a
Minecraft session.

Code is licensed under [Apache 2.0](LICENSE). The model layers derive from
Microsoft's TRELLIS implementation. CLIP is from OpenAI. Their license notices
are in [LICENSES](LICENSES). Model weights retain the licenses of their distribution.
