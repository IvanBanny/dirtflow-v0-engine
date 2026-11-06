"""Start a single inference worker."""

import argparse
import logging
import os
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Dirtflow model.")
    parser.add_argument(
        "--model", help="local bundle directory; defaults to the published model"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download and verify the model, then exit",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--resident", action="store_true", help="keep both model stages on the device"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sequential-guidance", action="store_true")
    parser.add_argument("--no-physics", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("threads must be positive and port must be in 1..65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.download_only:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu or --download-only")

    from . import bundle, download

    model = (
        pathlib.Path(args.model).expanduser() if args.model else download.fetch_model()
    )
    if args.download_only:
        logging.info("Verifying model bundle")
        bundle.validate_bundle(model)
        print(model)
        return

    import numpy as np
    import uvicorn

    from . import engine, sampling, server

    torch.set_num_threads(args.threads)
    logging.info("Loading model bundle")
    sampler = sampling.Sampler(str(model), args.device, offload=not args.resident)
    logging.info("Warming both stages")
    before = np.zeros((64, 64, 64), dtype=np.int64)
    edit = np.zeros_like(before, dtype=bool)
    edit[30:34, 30:34, 30:34] = True
    sampler.generate(
        before,
        edit,
        np.ones_like(edit),
        "a stone cube",
        [0],
        settings=sampling.Settings(1, 1, physics=False, preview_every=0),
    )
    runtime = engine.Engine(sampler, not args.no_physics, not args.sequential_guidance)
    app = server.create_app(runtime.generate, os.environ.get("DIRTFLOW_TOKEN") or None)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
