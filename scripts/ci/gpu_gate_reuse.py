#!/usr/bin/env python3
"""Decide whether a tested ref already has a successful paid CI GPU gate.

`runpod-gpu-test.yml` is triggered by `workflow_run` `in_progress`, which
GitHub delivers several times per upstream run (#2002 measured 3.0 RunPod
runs per upstream run). Every delivery used to provision its own pod for the
same pinned merge ref. The gate job now records the ref it validated in the
check run's `external_id` (`runpod-gpu-gate:v1:<tested ref>:<kind>`), and the
trusted jobs ask this script whether the newest gate for exactly that ref is
a paid success before spending again.

Only a successful gate whose kind is `paid` (the pod ran the tests) or
`reused` (a later delivery that reused such a result) is reusable. Gates
published for "GPU not required", a local GPU validation, a skipped paid
path, or a failure are never reusable, and neither are legacy gates without the marker. The newest
completed gate for the ref decides, so a later failure for the same ref
re-enables the paid path.

The script fails open toward validation: a missing input, an HTTP or parse
error, or an unexpected payload all answer `reuse=false`, which keeps the
paid path running exactly as before.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import Any

CHECK_NAME = "CI GPU gate"
CHECK_APP_SLUG = "github-actions"
MARKER_PREFIX = "runpod-gpu-gate:v1"
REUSABLE_KINDS = ("paid", "reused")
GATE_KINDS = ("paid", "reused", "failed", "not-required", "local", "skipped")
PAGE_SIZE = 100
MAX_PAGES = 10

Transport = Callable[[str], tuple[int, bytes]]


class ReuseLookupError(RuntimeError):
    """The check-run listing could not be read or had an unexpected shape."""


def gate_external_id(tested_ref: str, kind: str) -> str:
    """Return the `external_id` marker the gate publishes for `tested_ref`."""

    if kind not in GATE_KINDS:
        raise ValueError(f"unknown CI GPU gate kind: {kind}")
    if not tested_ref:
        raise ValueError("tested ref must not be empty")
    return f"{MARKER_PREFIX}:{tested_ref}:{kind}"


def _marker_kind(check: Mapping[str, Any], tested_ref: str) -> str | None:
    external_id = check.get("external_id")
    prefix = f"{MARKER_PREFIX}:{tested_ref}:"
    if not isinstance(external_id, str) or not external_id.startswith(prefix):
        return None
    kind = external_id[len(prefix) :]
    return kind if kind in GATE_KINDS else None


def find_reusable_gate(
    check_runs: Iterable[Mapping[str, Any]], tested_ref: str
) -> Mapping[str, Any] | None:
    """Return the newest gate for `tested_ref` if it is a reusable success."""

    if not tested_ref:
        return None
    gates = []
    for check in check_runs:
        if check.get("name") != CHECK_NAME:
            continue
        app = check.get("app")
        if not isinstance(app, Mapping) or app.get("slug") != CHECK_APP_SLUG:
            continue
        if check.get("status") != "completed":
            continue
        kind = _marker_kind(check, tested_ref)
        if kind is None:
            continue
        completed_at = check.get("completed_at")
        check_id = check.get("id")
        if not isinstance(completed_at, str) or not isinstance(check_id, int):
            continue
        gates.append((completed_at, check_id, kind, check))
    if not gates:
        return None
    # ISO-8601 UTC timestamps from the Checks API sort lexicographically; the
    # check id breaks ties between gates completed in the same second.
    _, _, kind, newest = max(gates, key=lambda gate: (gate[0], gate[1]))
    if newest.get("conclusion") == "success" and kind in REUSABLE_KINDS:
        return newest
    return None


def list_gate_check_runs(
    transport: Transport, repository: str, head_sha: str
) -> list[Mapping[str, Any]]:
    """List every `CI GPU gate` check run on `head_sha` (not only the latest)."""

    checks: list[Mapping[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        query = urllib.parse.urlencode(
            {
                "check_name": CHECK_NAME,
                # The default `latest` filter hides older gates, including the
                # one for this ref when a gate for another base came later.
                "filter": "all",
                "per_page": PAGE_SIZE,
                "page": page,
            }
        )
        url = (
            f"https://api.github.com/repos/{repository}/commits/{head_sha}"
            f"/check-runs?{query}"
        )
        status, body = transport(url)
        if status != 200:
            raise ReuseLookupError(f"GET {url} returned HTTP {status}")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise ReuseLookupError(f"GET {url} returned invalid JSON: {error}") from error
        runs = payload.get("check_runs") if isinstance(payload, Mapping) else None
        if not isinstance(runs, list):
            raise ReuseLookupError(f"GET {url} has no 'check_runs' array")
        checks.extend(run for run in runs if isinstance(run, Mapping))
        total = payload.get("total_count")
        if len(runs) < PAGE_SIZE or (isinstance(total, int) and len(checks) >= total):
            return checks
    raise ReuseLookupError(f"more than {MAX_PAGES * PAGE_SIZE} gate check runs")


def _github_transport(token: str) -> Transport:
    def send(url: str) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tenferro-ci-gpu-gate-reuse/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    return send


def decide(
    *,
    repository: str,
    head_sha: str,
    tested_ref: str,
    transport: Transport | None,
    checks_json: str | None = None,
) -> Mapping[str, Any] | None:
    """Return the reusable gate, or None. Never raises: errors mean "run"."""

    if not head_sha or not tested_ref:
        print("No PR head or tested ref; the paid path runs.")
        return None
    try:
        if checks_json is not None:
            with open(checks_json, encoding="utf-8") as handle:
                payload = json.load(handle)
            runs = payload.get("check_runs") if isinstance(payload, Mapping) else payload
            if not isinstance(runs, list):
                raise ReuseLookupError(f"{checks_json} has no check run list")
            checks = [run for run in runs if isinstance(run, Mapping)]
        else:
            if transport is None:
                raise ReuseLookupError("no GitHub token for the check-run lookup")
            checks = list_gate_check_runs(transport, repository, head_sha)
    except (OSError, ValueError, ReuseLookupError) as error:
        print(f"::warning::CI GPU gate reuse lookup failed; the paid path runs: {error}")
        return None
    reusable = find_reusable_gate(checks, tested_ref)
    if reusable is None:
        print(f"No reusable successful CI GPU gate for {tested_ref}; the paid path runs.")
    else:
        print(
            f"Tested ref {tested_ref} already passed the paid CI GPU gate: "
            f"{reusable.get('html_url')} (completed {reusable.get('completed_at')})."
        )
    return reusable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--tested-ref", default="")
    parser.add_argument(
        "--checks-json",
        help="Read check runs from this file instead of the API (dry run)",
    )
    parser.add_argument(
        "--output",
        help="Append GitHub-output style key=value lines to this file",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token = os.environ.get("GH_TOKEN", "")
    reusable = decide(
        repository=args.repository,
        head_sha=args.head_sha,
        tested_ref=args.tested_ref,
        transport=_github_transport(token) if token else None,
        checks_json=args.checks_json,
    )
    lines = ["reuse=false", "reused_check_url="]
    if reusable is not None:
        url = reusable.get("html_url") or reusable.get("details_url") or ""
        lines = ["reuse=true", f"reused_check_url={url}"]
    for line in lines:
        print(line)
    if args.output:
        with open(args.output, "a", encoding="utf-8") as output:
            output.write("".join(f"{line}\n" for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
