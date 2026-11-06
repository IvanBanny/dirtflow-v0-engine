"""HTTP v2 job API at the mod's /v1 URLs."""

import contextlib
import hmac
import json
import time
import zlib
from collections.abc import AsyncIterator, Callable

import fastapi
from sse_starlette import sse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import jobs
from .request import Request


class BodyLimit:
    def __init__(self, app: ASGIApp, limit: int = 32 * 1024 * 1024) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Buffer at most the request budget before JSON parsing
        chunks, length = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            length += len(body)
            if length > self.limit:
                await fastapi.responses.JSONResponse(
                    {"detail": "request exceeds byte limit"}, status_code=413
                )(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def buffered() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": b"".join(chunks),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, buffered, send)


async def stream(job: jobs.Job, cursor: int) -> AsyncIterator[dict]:
    while True:
        for sequence, kind, data in job.since(cursor):
            if kind in {"queue", "preview", "progress"} and (
                job.finished is not None or job.latest.get(kind, (None,))[0] != sequence
            ):
                continue
            cursor = sequence
            yield {
                "id": str(sequence),
                "event": "status" if kind == "queue" else kind,
                "data": json.dumps(data, separators=(",", ":")),
            }
        if job.finished is not None:
            if job.since(cursor):
                continue
            return
        job.changed.clear()
        if not job.since(cursor):
            await job.changed.wait()


class Events(sse.EventSourceResponse):
    def __init__(self, job: jobs.Job, cursor: int) -> None:
        self.job = job
        super().__init__(stream(job, cursor), ping=10, send_timeout=10)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job.subscribers -= 1


def create_app(
    generate: Callable[[jobs.Job, jobs.Mailbox], dict],
    token: str | None = None,
    max_waiting: int = 8,
) -> fastapi.FastAPI:
    queue = jobs.Queue(generate, max_waiting)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        queue.start()
        try:
            yield
        finally:
            await queue.stop()

    app = fastapi.FastAPI(title="Dirtflow", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)
    app.state.queue = queue

    def auth(req: fastapi.Request) -> None:
        supplied = req.headers.get("authorization", "").encode()
        if token is not None and not hmac.compare_digest(
            supplied, ("Bearer " + token).encode()
        ):
            raise fastapi.HTTPException(401, "invalid bearer token")

    def lookup(job_id: str) -> jobs.Job:
        queue.prune()
        if job_id not in queue.jobs:
            raise fastapi.HTTPException(404, "unknown or expired job")
        return queue.jobs[job_id]

    @app.get("/v1/health")
    async def health() -> dict:
        return {"status": "ready", "protocol_version": 2}

    @app.post("/v1/generate", status_code=202, dependencies=[fastapi.Depends(auth)])
    async def generate_job(req: Request) -> dict:
        try:
            tensor = req.decode()
        except (ValueError, zlib.error) as error:
            raise fastapi.HTTPException(422, str(error)) from error
        job = jobs.Job(req, tensor)
        try:
            position = queue.submit(job)
        except jobs.QueueFull as error:
            raise fastapi.HTTPException(429, "generation queue is full") from error
        return {"job_id": job.id, "status": "queued", "queue_position": position}

    @app.get("/v1/jobs/{job_id}", dependencies=[fastapi.Depends(auth)])
    async def status(job_id: str) -> dict:
        job = lookup(job_id)
        return {
            "job_id": job.id,
            "phase": job.phase,
            "elapsed_ms": int(
                ((job.finished or time.monotonic()) - job.submitted) * 1000
            ),
        }

    @app.delete(
        "/v1/jobs/{job_id}", status_code=204, dependencies=[fastapi.Depends(auth)]
    )
    async def cancel(job_id: str) -> None:
        queue.cancel(lookup(job_id))

    @app.get("/v1/jobs/{job_id}/events", dependencies=[fastapi.Depends(auth)])
    async def events(job_id: str, req: fastapi.Request) -> Events:
        job = lookup(job_id)
        try:
            cursor = int(req.headers.get("last-event-id", "0"))
            if not 0 <= cursor <= job.sequence:
                raise ValueError
        except ValueError as error:
            raise fastapi.HTTPException(400, "invalid Last-Event-ID") from error
        if job.subscribers >= 2:
            raise fastapi.HTTPException(429, "too many subscribers")
        job.subscribers += 1
        return Events(job, cursor)

    return app
