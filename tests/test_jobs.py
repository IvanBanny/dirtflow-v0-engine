import asyncio
import threading

import pytest

from dirtflow_engine import jobs
from dirtflow_engine.request import Request


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def make_job(data):
    req = Request.model_validate(data)
    return jobs.Job(req, req.decode())


async def test_serial_queue_cancel_and_shutdown(request_data):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def generate(job, mailbox):
        calls.append(job.id)
        entered.set()
        assert release.wait(5)
        return {"seed": 1}

    queue = jobs.Queue(generate, max_waiting=1)
    queue.start()
    first, second, third = [make_job(request_data) for _ in range(3)]
    queue.submit(first)
    await until(entered.is_set)
    assert queue.submit(second) == 1
    with pytest.raises(jobs.QueueFull):
        queue.submit(third)
    queue.cancel(second)
    assert second.phase == "CANCELED"
    queue.submit(third)
    queue.cancel(first)
    assert queue.active is first
    shutdown = asyncio.create_task(queue.stop())
    await asyncio.sleep(0.03)
    assert not shutdown.done()
    release.set()
    await shutdown
    assert calls == [first.id]
    assert all(j.phase == "CANCELED" for j in (first, second, third))
    assert first.tensor is None
    assert first.request.snapshot.data == ""


async def test_worker_recovers_and_retention_is_bounded(request_data):
    calls = 0

    def generate(job, mailbox):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("bad model input")
        for i in range(100):
            mailbox.put("preview", frame_sequence=i)
        return {"seed": 2}

    queue = jobs.Queue(generate, retention=1)
    queue.start()
    first = make_job(request_data)
    queue.submit(first)
    await until(lambda: first.finished is not None)
    assert first.phase == "FAILED"
    second = make_job(request_data)
    queue.submit(second)
    await until(lambda: second.finished is not None)
    await queue.stop()
    assert second.phase == "COMPLETE"
    assert len(queue.jobs) == 1
    assert [e[1] for e in second.controls] == ["status", "status", "result", "end"]
    assert not second.latest


def test_latest_events_are_replaceable(request_data):
    job = make_job(request_data)
    for i in range(1000):
        job.push("preview", frame_sequence=i)
        job.push("queue", queue_position=i)
    assert len(job.since(0)) == 2
    job.finish("COMPLETE", "result", seed=0)
    assert [e[1] for e in job.since(0)] == ["result", "end"]
