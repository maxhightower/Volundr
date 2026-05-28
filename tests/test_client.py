import pytest

from volundr.invoke.client import InvokeAIClient, InvokeAIError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json


class FakeSession:
    """Records requests and returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._responses.pop(0)


def test_enqueue_posts_graph_and_returns_batch_id():
    session = FakeSession([FakeResponse(json_data={"batch": {"batch_id": "b-123"}})])
    client = InvokeAIClient(base_url="http://host:9090", session=session)

    batch_id = client.enqueue({"nodes": {}, "edges": []})

    assert batch_id == "b-123"
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "http://host:9090/api/v1/queue/default/enqueue_batch"
    assert call["json"]["batch"]["graph"] == {"nodes": {}, "edges": []}
    assert call["json"]["batch"]["runs"] == 1


def test_enqueue_raises_without_batch_id():
    session = FakeSession([FakeResponse(json_data={"unexpected": True})])
    client = InvokeAIClient(session=session)
    with pytest.raises(InvokeAIError):
        client.enqueue({})


def test_http_error_raised():
    session = FakeSession([FakeResponse(status_code=500, text="boom")])
    client = InvokeAIClient(session=session)
    with pytest.raises(InvokeAIError):
        client.enqueue({})


def test_wait_for_batch_completes():
    session = FakeSession(
        [FakeResponse(json_data={"total": 1, "completed": 1, "pending": 0, "in_progress": 0})]
    )
    client = InvokeAIClient(session=session)
    status = client.wait_for_batch("b-1", poll_interval=0, timeout=5)
    assert status["completed"] == 1


def test_wait_for_batch_raises_on_failure():
    session = FakeSession([FakeResponse(json_data={"total": 1, "failed": 1})])
    client = InvokeAIClient(session=session)
    with pytest.raises(InvokeAIError):
        client.wait_for_batch("b-1", poll_interval=0, timeout=5)


def test_wait_for_batch_times_out():
    session = FakeSession(
        [FakeResponse(json_data={"total": 1, "completed": 0, "pending": 1})]
    )
    client = InvokeAIClient(session=session)
    with pytest.raises(InvokeAIError):
        client.wait_for_batch("b-1", poll_interval=0, timeout=0)


def test_image_url():
    client = InvokeAIClient(base_url="http://host:9090")
    assert client.image_url("img.png") == "http://host:9090/api/v1/images/i/img.png/full"


def test_download_image_writes_file(tmp_path):
    session = FakeSession([FakeResponse(content=b"PNGDATA")])
    client = InvokeAIClient(session=session)
    dest = tmp_path / "out.png"
    client.download_image("img.png", str(dest))
    assert dest.read_bytes() == b"PNGDATA"
