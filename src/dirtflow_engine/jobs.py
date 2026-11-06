"""One sampling thread, a bounded waiting queue, and reconnectable job events."""

import asyncio
import collections
import dataclasses
import logging
import threading
import time
import uuid

from .request import Request
from .voxel import Tensor


class QueueFull(Exception):
    """The waiting queue has reached its limit."""


@dataclasses.dataclass
class Job:
    request: Request
    tensor: Tensor | None
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    phase: str = "QUEUED"
    submitted: float = dataclasses.field(default_factory=time.monotonic)
    finished: float | None = None
    canceled: threading.Event = dataclasses.field(default_factory=threading.Event)
    changed: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    sequence: int = 0
    controls: list = dataclasses.field(default_factory=list)
    latest: dict = dataclasses.field(default_factory=dict)
    subscribers: int = 0

    def push(self, kind: str, **data) -> None:
        if self.finished is not None:
            return
        self.sequence += 1
        event = (self.sequence, kind, {"job_id": self.id, **data})
        if kind in {"preview", "progress", "queue"}:
            self.latest[kind] = event
        else:
            self.controls.append(event)
        self.changed.set()

    def finish(self, phase: str, kind: str, **data) -> None:
        if self.finished is not None:
            return
        self.latest.clear()
        self.phase = phase
        self.push(kind, **data)
        self.push("end", phase=phase)
        self.finished = time.monotonic()
        self.tensor = None
        # The request's encoded snapshot can be several MiB
        self.request.snapshot.data = ""

    def since(self, cursor: int) -> list:
        return sorted(
            event
            for event in [*self.controls, *self.latest.values()]
            if event[0] > cursor
        )


class Mailbox:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: dict[str, dict] = {}

    def put(self, kind: str, **data) -> None:
        with self.lock:
            self.frames[kind] = data

    def take(self) -> dict:
        with self.lock:
            frames, self.frames = self.frames, {}
            return frames


class Queue:
    def __init__(
        self, generate, max_waiting: int = 8, retention: int = 32, ttl: float = 900
    ) -> None:
        if min(max_waiting, retention, ttl) <= 0:
            raise ValueError("queue and retention limits must be positive")
        self.generate = generate
        self.max_waiting, self.retention, self.ttl = max_waiting, retention, ttl
        self.jobs: dict[str, Job] = {}
        self.pending: collections.deque[Job] = collections.deque()
        self.available = asyncio.Event()
        self.active: Job | None = None
        self.worker: asyncio.Task | None = None
        self.closing = False

    def start(self) -> None:
        self.worker = asyncio.create_task(self.run())

    def prune(self) -> None:
        done = sorted(
            (
                job
                for job in self.jobs.values()
                if job.finished is not None and job is not self.active
            ),
            key=lambda job: job.finished or 0,
        )
        for i, job in enumerate(done):
            if (
                i < len(done) - self.retention
                or time.monotonic() - (job.finished or 0) > self.ttl
            ):
                del self.jobs[job.id]

    def submit(self, job: Job) -> int:
        self.prune()
        if self.closing or len(self.pending) >= self.max_waiting:
            raise QueueFull
        position = len(self.pending) + int(self.active is not None)
        self.jobs[job.id] = job
        self.pending.append(job)
        job.push("status", phase="QUEUED", queue_position=position)
        self.available.set()
        return position

    def positions(self) -> None:
        for position, job in enumerate(
            self.pending, start=int(self.active is not None)
        ):
            job.push("queue", phase="QUEUED", queue_position=position)

    def cancel(self, job: Job) -> None:
        if job.finished is not None:
            return
        job.canceled.set()
        if job is not self.active:
            self.pending.remove(job)
            job.finish(
                "CANCELED", "error", code="CANCELED", message="canceled by client"
            )
        self.positions()

    async def stop(self) -> None:
        self.closing = True
        for job in list(self.pending):
            self.cancel(job)
        if self.active is not None:
            self.cancel(self.active)
        self.available.set()
        if self.worker is not None:
            await self.worker

    async def run(self) -> None:
        while True:
            if not self.pending:
                if self.closing:
                    return
                self.available.clear()
                await self.available.wait()
                continue
            job = self.pending.popleft()
            self.active = job
            job.phase = "GENERATING"
            job.latest.pop("queue", None)
            job.push("status", phase=job.phase)
            self.positions()
            mailbox = Mailbox()
            task = asyncio.create_task(asyncio.to_thread(self.generate, job, mailbox))
            try:
                while not task.done():
                    for kind, data in mailbox.take().items():
                        if not job.canceled.is_set():
                            job.push(kind, **data)
                    await asyncio.sleep(0.02)
                result = task.result()
                if not job.canceled.is_set():
                    job.finish("COMPLETE", "result", **result)
            except asyncio.CancelledError:
                job.canceled.set()
                raise
            except Exception as error:
                if not job.canceled.is_set():
                    logging.exception("generation failed for %s", job.id)
                    job.finish(
                        "FAILED", "error", code="MODEL_FAILURE", message=str(error)
                    )
            finally:
                # A canceled await cannot stop the sampling thread
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        job.canceled.set()
                    except Exception:
                        break
                if not task.cancelled():
                    task.exception()
                if job.canceled.is_set():
                    job.finish(
                        "CANCELED",
                        "error",
                        code="CANCELED",
                        message="generation canceled",
                    )
                self.active = None
                self.positions()
                self.prune()
