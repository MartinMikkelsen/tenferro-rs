#!/usr/bin/env python3
"""Report a stale or deprecated pinned GitHub Actions runner for RunPod pods.

RunPod pods register a JIT runner from the release pinned in
`.github/workflows/runpod-gpu-execute.yml` (`RUNNER_VERSION` and
`RUNNER_SHA256`). GitHub stops queueing jobs to runners that fall too far
behind, and a rejected runner is only noticed when a paid provision ladder
fails: pods start, pass the CUDA smoke proof, and never register (#1921,
2026-09-24..26). This check compares the pin against `actions/runner`
releases so the next deprecation is reported by a free scheduled job.

Policy (conservative; the exact GitHub rule is not published per release):

- the pin must be a published, non-prerelease release whose `linux-x64`
  checksum in the release notes equals `RUNNER_SHA256`;
- a newer release makes the check warn;
- it fails once a newer release is `--max-lag-days` old (default 14), or two
  or more newer releases exist. Observed: 2.335.1 kept working for 66 days
  after 2.336.0, and was rejected 29 days after 2.337.0 shipped, so both
  conditions fire well before that.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

PIN_FILE = Path(".github/workflows/runpod-gpu-execute.yml")
RELEASES_URL = "https://api.github.com/repos/actions/runner/releases?per_page=100"
DEFAULT_MAX_LAG_DAYS = 14

_VERSION_PIN = re.compile(r'^\s*RUNNER_VERSION="(?P<value>[0-9]+\.[0-9]+\.[0-9]+)"\s*$', re.M)
_SHA_PIN = re.compile(r'^\s*RUNNER_SHA256="(?P<value>[0-9a-f]{64})"\s*$', re.M)
_RELEASE_SHA = re.compile(
    r"<!-- BEGIN SHA linux-x64 -->(?P<value>[0-9a-f]{64})<!-- END SHA linux-x64 -->"
)

Fetch = Callable[[str], Any]


class PinCheckError(RuntimeError):
    """The pin or the release listing could not be read."""


def read_pin(text: str) -> tuple[str, str]:
    """Return (version, sha256) pinned in the workflow text."""

    versions = _VERSION_PIN.findall(text)
    shas = _SHA_PIN.findall(text)
    if len(versions) != 1 or len(shas) != 1:
        raise PinCheckError(
            f"expected exactly one RUNNER_VERSION and RUNNER_SHA256 pin, found "
            f"{len(versions)} and {len(shas)}"
        )
    return versions[0], shas[0]


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _release_version(release: Mapping[str, Any]) -> str | None:
    tag = release.get("tag_name")
    if not isinstance(tag, str) or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        return None
    return tag[1:]


def _published(release: Mapping[str, Any]) -> datetime.datetime:
    value = release.get("published_at")
    if not isinstance(value, str):
        raise PinCheckError(f"release {release.get('tag_name')} has no published_at")
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def evaluate(
    pin_version: str,
    pin_sha: str,
    releases: Iterable[Mapping[str, Any]],
    now: datetime.datetime,
    max_lag_days: int = DEFAULT_MAX_LAG_DAYS,
) -> tuple[str, list[str]]:
    """Return ("ok" | "warn" | "fail", messages) for the pin."""

    stable = [
        release
        for release in releases
        if not release.get("draft")
        and not release.get("prerelease")
        and _release_version(release) is not None
    ]
    by_version = {_release_version(release): release for release in stable}
    messages: list[str] = []
    pinned = by_version.get(pin_version)
    if pinned is None:
        return "fail", [f"Pinned runner {pin_version} is not a published stable actions/runner release."]
    body = pinned.get("body") if isinstance(pinned.get("body"), str) else ""
    published_sha = _RELEASE_SHA.search(body)
    if published_sha is None:
        messages.append(f"::warning::Release v{pin_version} notes carry no linux-x64 SHA-256 to compare.")
    elif published_sha.group("value") != pin_sha:
        return "fail", [
            f"RUNNER_SHA256 {pin_sha} does not match the published linux-x64 "
            f"checksum {published_sha.group('value')} of v{pin_version}."
        ]

    newer = sorted(
        (
            release
            for version, release in by_version.items()
            if version is not None and _version_key(version) > _version_key(pin_version)
        ),
        key=_published,
    )
    if not newer:
        messages.append(f"Pinned runner {pin_version} is the latest stable actions/runner release.")
        return "ok", messages
    latest = _release_version(newer[-1])
    oldest_newer = newer[0]
    lag_days = (now - _published(oldest_newer)).total_seconds() / 86400.0
    summary = (
        f"Pinned runner {pin_version} is behind {len(newer)} newer release(s) "
        f"(latest v{latest}); v{_release_version(oldest_newer)} was published "
        f"{lag_days:.0f} day(s) ago."
    )
    if len(newer) >= 2 or lag_days >= max_lag_days:
        messages.append(
            f"{summary} Bump RUNNER_VERSION and RUNNER_SHA256 in {PIN_FILE} "
            "before GitHub stops queueing jobs to it (see "
            "docs/design/runpod-gpu-provisioning.md, Runner pin runbook)."
        )
        return "fail", messages
    messages.append(f"::warning::{summary} Bump the pin within {max_lag_days} days of that release.")
    return "warn", messages


def _fetch_json(url: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "tenferro-ci-runner-pin-check/1",
    }
    if token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise PinCheckError(f"GET {url} failed: {error}") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-file", type=Path, default=PIN_FILE)
    parser.add_argument(
        "--releases-json",
        type=Path,
        help="Read the actions/runner release list from this file (offline dry run)",
    )
    parser.add_argument("--max-lag-days", type=int, default=DEFAULT_MAX_LAG_DAYS)
    parser.add_argument("--now", help="ISO-8601 time to evaluate at (tests)")
    return parser.parse_args()


def main(fetch: Fetch = _fetch_json) -> int:
    args = _parse_args()
    try:
        version, sha = read_pin(args.pin_file.read_text(encoding="utf-8"))
        if args.releases_json is not None:
            releases = json.loads(args.releases_json.read_text(encoding="utf-8"))
        else:
            releases = fetch(RELEASES_URL)
        if not isinstance(releases, list):
            raise PinCheckError("release listing is not a JSON array")
    except (OSError, ValueError, PinCheckError) as error:
        print(f"::error::Runner pin check could not run: {error}")
        return 1
    now = (
        datetime.datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        if args.now
        else datetime.datetime.now(datetime.timezone.utc)
    )
    verdict, messages = evaluate(
        version, sha, (r for r in releases if isinstance(r, Mapping)), now, args.max_lag_days
    )
    for message in messages:
        prefix = "::error::" if verdict == "fail" and not message.startswith("::") else ""
        print(f"{prefix}{message}")
    print(f"runner_pin={version} verdict={verdict}")
    return 1 if verdict == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
