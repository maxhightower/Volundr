"""Graph-agnostic REST client for InvokeAI's queue API.

Deliberately knows nothing about graph *contents* — it enqueues whatever batch
dict it's given, polls the queue item to completion, and resolves output images.
That keeps it decoupled from InvokeAI's per-version node schema (which must be
validated against a live instance on a GPU box; see graph.py).

Uses polling rather than the Socket.IO progress channel to avoid a websocket
dependency; swap in socketio later if live step previews are wanted.

NOTE: endpoint paths follow InvokeAI's documented v1 queue API. They have
shifted across major versions — confirm against the target instance's
`/openapi.json` before relying on them in production.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import requests


class InvokeAIError(RuntimeError):
    pass


class _HttpSession(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class InvokeAIClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9090",
        queue_id: str = "default",
        session: _HttpSession | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.queue_id = queue_id
        self.session = session or requests.Session()
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self.session.request(
            method, self._url(path), timeout=self.timeout, **kwargs
        )
        if resp.status_code >= 400:
            raise InvokeAIError(f"{method} {path} -> {resp.status_code}: {resp.text}")
        return resp.json()

    def enqueue(self, graph: dict) -> str:
        """Enqueue a single graph as a batch; return the enqueued batch id."""
        payload = {"prepend": False, "batch": {"graph": graph, "runs": 1}}
        data = self._json(
            "POST",
            f"/api/v1/queue/{self.queue_id}/enqueue_batch",
            json=payload,
        )
        batch_id = data.get("batch", {}).get("batch_id") or data.get("batch_id")
        if not batch_id:
            raise InvokeAIError(f"no batch_id in enqueue response: {data}")
        return batch_id

    def batch_status(self, batch_id: str) -> dict:
        return self._json(
            "GET",
            f"/api/v1/queue/{self.queue_id}/b/{batch_id}/status",
        )

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
    ) -> dict:
        """Block until the batch finishes; return its final status.

        Raises on timeout or if any item in the batch fails.
        """
        deadline = time.monotonic() + timeout
        while True:
            status = self.batch_status(batch_id)
            if status.get("failed", 0) or status.get("canceled", 0):
                raise InvokeAIError(f"batch {batch_id} did not complete: {status}")
            pending = status.get("pending", 0) + status.get("in_progress", 0)
            if pending == 0 and status.get("completed", 0) >= status.get("total", 0):
                return status
            if time.monotonic() >= deadline:
                raise InvokeAIError(f"batch {batch_id} timed out: {status}")
            time.sleep(poll_interval)

    def image_url(self, image_name: str) -> str:
        return self._url(f"/api/v1/images/i/{image_name}/full")

    def download_image(self, image_name: str, dest: str) -> str:
        resp = self.session.request(
            "GET", self.image_url(image_name), timeout=self.timeout
        )
        if resp.status_code >= 400:
            raise InvokeAIError(f"download {image_name} -> {resp.status_code}")
        with open(dest, "wb") as fh:
            fh.write(resp.content)
        return dest
