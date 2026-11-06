from fastapi.testclient import TestClient

from dirtflow_engine import server, voxel


def result(job, mailbox):
    diff = voxel.Tensor(
        tuple(job.request.region.min), job.request.region.size, [voxel.AIR], [0], [0]
    )
    return {"diff": voxel.pack(diff), "seed": 42}


def test_http_lifecycle_and_reconnect(request_data):
    with TestClient(server.create_app(result, token="test")) as client:
        assert client.get("/v1/health").json()["protocol_version"] == 2
        assert client.post("/v1/generate", json=request_data).status_code == 401
        headers = {"Authorization": "Bearer test"}
        response = client.post("/v1/generate", json=request_data, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        url = f"/v1/jobs/{job_id}"
        stream = client.get(url + "/events", headers=headers)
        assert "event: result" in stream.text and "event: end" in stream.text
        assert '"seed":42' in stream.text
        status = client.get(url, headers=headers).json()
        assert status["phase"] == "COMPLETE"
        sequence = client.app.state.queue.jobs[job_id].sequence
        assert (
            client.get(
                url + "/events", headers=headers | {"Last-Event-ID": str(sequence)}
            ).text
            == ""
        )
        assert (
            client.get(
                url + "/events", headers=headers | {"Last-Event-ID": "-1"}
            ).status_code
            == 400
        )
        assert client.delete(url, headers=headers).status_code == 204
        assert client.app.state.queue.jobs[job_id].subscribers == 0


def test_http_validation(request_data):
    with TestClient(server.create_app(result)) as client:
        request_data["snapshot"]["data"] = "!!!!"
        assert client.post("/v1/generate", json=request_data).status_code == 422
        assert client.get("/v1/jobs/missing").status_code == 404
        assert (
            client.post(
                "/v1/generate", content=b"x" * (32 * 1024 * 1024 + 1)
            ).status_code
            == 413
        )
