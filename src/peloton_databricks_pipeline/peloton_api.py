from __future__ import annotations

from datetime import datetime
from typing import Any

import requests


class PelotonClient:
    """Minimal Peloton client using Peloton's web API endpoints."""

    def __init__(self, username: str, password: str, base_url: str = "https://api.onepeloton.com") -> None:
        self.username = username
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def authenticate(self) -> None:
        payload = {"username_or_email": self.username, "password": self.password}
        response = self.session.post(f"{self.base_url}/auth/login", json=payload, timeout=30)
        response.raise_for_status()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_me(self) -> dict[str, Any]:
        return self._get("/api/me")

    def get_workouts(self, page_size: int = 50) -> list[dict[str, Any]]:
        user_id = self.get_me()["id"]
        page = 0
        all_workouts: list[dict[str, Any]] = []

        while True:
            params = {"joins": "ride,ride.instructor", "limit": page_size, "page": page, "sort_by": "-created"}
            payload = self._get(f"/api/user/{user_id}/workouts", params=params)
            data = payload.get("data", [])
            if not data:
                break

            all_workouts.extend(data)
            page_count = payload.get("page_count", 0)
            page += 1
            if page >= page_count:
                break

        return all_workouts

    def get_workout_performance(self, workout_id: str) -> dict[str, Any]:
        return self._get(f"/api/workout/{workout_id}/performance_graph", params={"every_n": 5})


def filter_workouts_since(workouts: list[dict[str, Any]], since: str | None) -> list[dict[str, Any]]:
    if not since:
        return workouts

    since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
    filtered: list[dict[str, Any]] = []
    for workout in workouts:
        created_at = workout.get("created_at")
        if created_at is None:
            continue

        created_dt = datetime.fromtimestamp(created_at, tz=since_dt.tzinfo)
        if created_dt >= since_dt:
            filtered.append(workout)

    return filtered
