from __future__ import annotations

import base64
import hashlib
import html
import os
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests


class PelotonAuthError(RuntimeError):
    """Raised when Peloton authentication fails."""


class PelotonClient:
    """Peloton client using current OAuth-based web API auth flow."""

    AUTH_DOMAIN = "auth.onepeloton.com"
    AUTH_CLIENT_ID = "WVoJxVDdPoFx4RNewvvg6ch2mZ7bwnsM"
    AUTH_AUDIENCE = "https://api.onepeloton.com/"
    AUTH_SCOPE = "offline_access openid peloton-api.members:default"
    AUTH_REDIRECT_URI = "https://members.onepeloton.com/callback"
    AUTH0_CLIENT_PAYLOAD = "eyJuYW1lIjoiYXV0aDAuanMtdWxwIiwidmVyc2lvbiI6IjkuMTQuMyJ9"
    AUTH_AUTHORIZE_PATH = "/authorize"
    AUTH_TOKEN_PATH = "/oauth/token"

    def __init__(self, username: str, password: str, base_url: str = "https://api.onepeloton.com") -> None:
        self.username = username
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Peloton-Platform": "web"})
        self._access_token: str | None = None

    def authenticate(self) -> None:
        code_verifier = self._generate_random_string(64)
        code_challenge = self._generate_code_challenge(code_verifier)
        state = self._generate_random_string(32)
        nonce = self._generate_random_string(32)

        authorize_url = self._build_authorize_url(
            code_challenge=code_challenge,
            state=state,
            nonce=nonce,
        )

        auth_response = self.session.get(authorize_url, allow_redirects=True, timeout=30)
        login_url = auth_response.url
        login_query = parse_qs(urlparse(login_url).query)
        state = login_query.get("state", [state])[0]
        nonce = login_query.get("nonce", [nonce])[0]
        code_challenge = login_query.get("code_challenge", [code_challenge])[0]
        csrf = self._get_cookie("_csrf")
        if not csrf:
            raise PelotonAuthError("Peloton OAuth flow did not return CSRF cookie.")

        next_url = self._submit_credentials(
            login_url=login_url,
            csrf_token=csrf,
            state=state,
            nonce=nonce,
            code_challenge=code_challenge,
        )
        auth_code = self._follow_redirects_for_code(next_url)
        token_payload = self._exchange_code_for_token(auth_code, code_verifier)

        access_token = token_payload.get("access_token")
        if not access_token:
            raise PelotonAuthError("Peloton token response missing access_token.")

        self._access_token = access_token
        self.session.headers.update({"Authorization": f"Bearer {access_token}"})
        # Validate auth and fail early with clear error.
        self.get_me()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        retriable_statuses = {429, 500, 502, 503, 504}
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            response = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
            if response.status_code < 400:
                return response.json()

            if response.status_code in retriable_statuses and attempt < max_attempts:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = 2 ** (attempt - 1)
                time.sleep(min(delay, 30))
                continue

            response.raise_for_status()

        raise RuntimeError("Peloton GET request failed without a specific exception.")

    def _generate_random_string(self, length: int) -> str:
        raw = base64.urlsafe_b64encode(os.urandom(length)).decode("utf-8")
        return raw.rstrip("=")[:length]

    def _generate_code_challenge(self, verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    def _build_authorize_url(self, code_challenge: str, state: str, nonce: str) -> str:
        params = {
            "client_id": self.AUTH_CLIENT_ID,
            "audience": self.AUTH_AUDIENCE,
            "scope": self.AUTH_SCOPE,
            "response_type": "code",
            "response_mode": "query",
            "redirect_uri": self.AUTH_REDIRECT_URI,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "auth0Client": self.AUTH0_CLIENT_PAYLOAD,
        }
        return f"https://{self.AUTH_DOMAIN}{self.AUTH_AUTHORIZE_PATH}?{urlencode(params)}"

    def _get_cookie(self, name: str) -> str | None:
        for cookie in self.session.cookies:
            if cookie.name == name:
                return cookie.value
        return None

    def _submit_credentials(
        self,
        login_url: str,
        csrf_token: str,
        state: str,
        nonce: str,
        code_challenge: str,
    ) -> str:
        payload = {
            "client_id": self.AUTH_CLIENT_ID,
            "redirect_uri": self.AUTH_REDIRECT_URI,
            "tenant": "peloton-prod",
            "response_type": "code",
            "scope": self.AUTH_SCOPE,
            "audience": self.AUTH_AUDIENCE,
            "_csrf": csrf_token,
            "state": state,
            "_intstate": "deprecated",
            "nonce": nonce,
            "username": self.username,
            "password": self.password,
            "connection": "pelo-user-password",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Origin": f"https://{self.AUTH_DOMAIN}",
            "Referer": login_url,
            "Auth0-Client": self.AUTH0_CLIENT_PAYLOAD,
        }

        login_endpoint = f"https://{self.AUTH_DOMAIN}/usernamepassword/login"
        response = self.session.post(
            login_endpoint,
            json=payload,
            headers=headers,
            allow_redirects=False,
            timeout=30,
        )

        location = response.headers.get("Location")
        if location:
            return self._ensure_absolute_url(location)

        # Some responses return an HTML page with hidden form fields to post next.
        action, hidden_fields = self._parse_hidden_form(response.text)
        action_url = self._ensure_absolute_url(action)

        form_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0",
        }
        form_response = self.session.post(
            action_url,
            data=hidden_fields,
            headers=form_headers,
            allow_redirects=True,
            timeout=30,
        )
        return form_response.headers.get("Location") or form_response.url

    def _ensure_absolute_url(self, maybe_relative: str) -> str:
        if maybe_relative.startswith("http://") or maybe_relative.startswith("https://"):
            return maybe_relative
        if maybe_relative.startswith("/"):
            return f"https://{self.AUTH_DOMAIN}{maybe_relative}"
        return f"https://{self.AUTH_DOMAIN}/{maybe_relative}"

    def _parse_hidden_form(self, html_text: str) -> tuple[str, dict[str, str]]:
        form_match = re.search(
            r"<form[^>]*action=[\"'](?P<action>[^\"']+)[\"'][^>]*>(?P<body>.*?)</form>",
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not form_match:
            snippet = (html_text or "")[:400]
            raise PelotonAuthError(f"Peloton login returned no redirect and no parsable hidden form: {snippet}")

        action = html.unescape(form_match.group("action"))
        body = form_match.group("body")
        fields: dict[str, str] = {}

        for tag_match in re.finditer(r"<input[^>]*>", body, flags=re.IGNORECASE):
            tag = tag_match.group(0)
            input_type = self._extract_html_attr(tag, "type")
            if (input_type or "").lower() != "hidden":
                continue
            name = self._extract_html_attr(tag, "name")
            value = self._extract_html_attr(tag, "value") or ""
            if name:
                fields[name] = html.unescape(value)

        return action, fields

    def _extract_html_attr(self, tag: str, attr: str) -> str | None:
        quoted = re.search(
            rf"{attr}\s*=\s*[\"'](?P<value>.*?)[\"']",
            tag,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if quoted:
            return quoted.group("value")

        unquoted = re.search(rf"{attr}\s*=\s*(?P<value>[^\s>]+)", tag, flags=re.IGNORECASE)
        if unquoted:
            return unquoted.group("value")
        return None

    def _follow_redirects_for_code(self, start_url: str) -> str:
        response = self.session.get(start_url, allow_redirects=True, timeout=30)
        candidates: list[str] = [start_url, response.url]

        if response.headers.get("Location"):
            candidates.append(response.headers["Location"])

        for hist in response.history:
            candidates.append(hist.url)
            if hist.headers.get("Location"):
                candidates.append(hist.headers["Location"])

        for candidate in candidates:
            parsed = urlparse(candidate)
            code = parse_qs(parsed.query).get("code", [None])[0]
            if code:
                return code

        body_snippet = response.text[:400]
        raise PelotonAuthError(f"Peloton OAuth callback missing authorization code: {body_snippet}")

    def _exchange_code_for_token(self, code: str, code_verifier: str) -> dict[str, Any]:
        endpoint = f"https://{self.AUTH_DOMAIN}{self.AUTH_TOKEN_PATH}"
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.AUTH_CLIENT_ID,
            "code_verifier": code_verifier,
            "code": code,
            "redirect_uri": self.AUTH_REDIRECT_URI,
        }
        response = self.session.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30,
        )
        if response.status_code >= 400:
            raise PelotonAuthError(f"Peloton token exchange failed: {response.text[:500]}")
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
