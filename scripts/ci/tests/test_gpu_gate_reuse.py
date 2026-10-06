import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.ci.gpu_gate_reuse import (
    CHECK_NAME,
    MAX_PAGES,
    PAGE_SIZE,
    ReuseLookupError,
    decide,
    find_reusable_gate,
    gate_external_id,
    list_gate_check_runs,
)

ROOT = Path(__file__).resolve().parents[3]
REPO = "tensor4all/tenferro-rs"
HEAD = "a" * 40
REF = "b" * 40
OTHER_REF = "c" * 40


def gate(
    check_id: int,
    *,
    ref: str | None = REF,
    kind: str = "paid",
    conclusion: str = "success",
    completed_at: str = "2026-10-04T10:00:00Z",
    status: str = "completed",
    name: str = CHECK_NAME,
    app: str = "github-actions",
) -> dict:
    return {
        "id": check_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "completed_at": completed_at,
        "external_id": gate_external_id(ref, kind) if ref else "",
        "app": {"slug": app},
        "html_url": f"https://github.com/{REPO}/runs/{check_id}",
    }


class FindReusableGateTests(unittest.TestCase):
    def test_paid_success_for_the_same_ref_is_reusable(self) -> None:
        found = find_reusable_gate([gate(1)], REF)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["id"], 1)

    def test_reused_success_chains(self) -> None:
        self.assertIsNotNone(find_reusable_gate([gate(1, kind="reused")], REF))

    def test_other_kinds_are_never_reusable(self) -> None:
        for kind in ("not-required", "local", "skipped", "failed"):
            with self.subTest(kind=kind):
                self.assertIsNone(find_reusable_gate([gate(1, kind=kind)], REF))

    def test_another_tested_ref_is_never_reusable(self) -> None:
        # Same head, moved base: a different merge ref was never tested.
        self.assertIsNone(find_reusable_gate([gate(1, ref=OTHER_REF)], REF))

    def test_legacy_and_foreign_checks_are_ignored(self) -> None:
        legacy = gate(1, ref=None)
        # Actions job checks carry a UUID external_id (seen on CI_gpu.yml).
        job_check = dict(gate(2), external_id="3f437d4b-527d-55de-8421-ede0b671445a")
        foreign_app = gate(3, app="some-other-app")
        renamed = gate(4, name="CI GPU gate (copy)")
        pending = gate(5, status="in_progress")
        self.assertIsNone(
            find_reusable_gate([legacy, job_check, foreign_app, renamed, pending], REF)
        )

    def test_newest_gate_for_the_ref_decides(self) -> None:
        older_success = gate(1, completed_at="2026-10-04T10:00:00Z")
        newer_failure = gate(
            2, kind="failed", conclusion="failure", completed_at="2026-10-04T11:00:00Z"
        )
        self.assertIsNone(find_reusable_gate([older_success, newer_failure], REF))
        newest_success = gate(3, completed_at="2026-10-04T12:00:00Z")
        found = find_reusable_gate([newer_failure, newest_success, older_success], REF)
        assert found is not None
        self.assertEqual(found["id"], 3)

    def test_same_second_ties_break_on_check_id(self) -> None:
        failure = gate(9, kind="failed", conclusion="failure")
        success = gate(8)
        self.assertIsNone(find_reusable_gate([success, failure], REF))

    def test_a_newer_gate_for_another_ref_does_not_hide_this_ref(self) -> None:
        other = gate(2, ref=OTHER_REF, kind="failed", conclusion="failure",
                     completed_at="2026-10-04T12:00:00Z")
        self.assertIsNotNone(find_reusable_gate([gate(1), other], REF))

    def test_empty_ref_never_reuses(self) -> None:
        self.assertIsNone(find_reusable_gate([gate(1)], ""))

    def test_external_id_rejects_unknown_kinds_and_empty_refs(self) -> None:
        with self.assertRaises(ValueError):
            gate_external_id(REF, "maybe")
        with self.assertRaises(ValueError):
            gate_external_id("", "paid")


class ListingTests(unittest.TestCase):
    def test_lists_every_gate_with_filter_all_and_pages(self) -> None:
        pages = {
            1: [gate(i) for i in range(PAGE_SIZE)],
            2: [gate(PAGE_SIZE + 1)],
        }
        urls: list[str] = []

        def transport(url: str) -> tuple[int, bytes]:
            urls.append(url)
            page = int(url.rsplit("page=", 1)[1])
            body = {"total_count": PAGE_SIZE + 1, "check_runs": pages[page]}
            return 200, json.dumps(body).encode()

        checks = list_gate_check_runs(transport, REPO, HEAD)
        self.assertEqual(len(checks), PAGE_SIZE + 1)
        self.assertEqual(len(urls), 2)
        self.assertIn(f"/repos/{REPO}/commits/{HEAD}/check-runs?", urls[0])
        self.assertIn("filter=all", urls[0])
        self.assertIn("check_name=CI+GPU+gate", urls[0])

    def test_http_and_shape_errors_raise(self) -> None:
        for status, body in ((500, b"{}"), (200, b"not json"), (200, b'{"x": 1}')):
            with self.subTest(status=status, body=body):
                with self.assertRaises(ReuseLookupError):
                    list_gate_check_runs(lambda url: (status, body), REPO, HEAD)

    def test_unbounded_listing_is_refused(self) -> None:
        full = json.dumps(
            {"total_count": 10**6, "check_runs": [gate(1)] * PAGE_SIZE}
        ).encode()
        calls = []

        def transport(url: str) -> tuple[int, bytes]:
            calls.append(url)
            return 200, full

        with self.assertRaises(ReuseLookupError):
            list_gate_check_runs(transport, REPO, HEAD)
        self.assertEqual(len(calls), MAX_PAGES)


class DecideFailsOpenTests(unittest.TestCase):
    def test_lookup_errors_mean_run(self) -> None:
        for transport in (
            lambda url: (403, b"{}"),
            lambda url: (200, b"garbage"),
            None,
        ):
            with self.subTest(transport=transport):
                self.assertIsNone(
                    decide(repository=REPO, head_sha=HEAD, tested_ref=REF, transport=transport)
                )

    def test_missing_inputs_mean_run(self) -> None:
        ok = lambda url: (200, json.dumps({"check_runs": [gate(1)]}).encode())
        self.assertIsNone(decide(repository=REPO, head_sha="", tested_ref=REF, transport=ok))
        self.assertIsNone(decide(repository=REPO, head_sha=HEAD, tested_ref="", transport=ok))
        self.assertIsNotNone(decide(repository=REPO, head_sha=HEAD, tested_ref=REF, transport=ok))

    def test_cli_dry_run_writes_github_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checks = Path(directory) / "checks.json"
            output = Path(directory) / "out.txt"
            for payload, expected in (
                ({"check_runs": [gate(7)]}, ["reuse=true", f"reused_check_url=https://github.com/{REPO}/runs/7"]),
                ({"check_runs": [gate(7, ref=OTHER_REF)]}, ["reuse=false", "reused_check_url="]),
                ("not a listing", ["reuse=false", "reused_check_url="]),
            ):
                with self.subTest(payload=payload):
                    checks.write_text(json.dumps(payload))
                    output.write_text("")
                    result = subprocess.run(
                        [sys.executable, "scripts/ci/gpu_gate_reuse.py",
                         "--repository", REPO, "--head-sha", HEAD, "--tested-ref", REF,
                         "--checks-json", str(checks), "--output", str(output)],
                        cwd=ROOT, capture_output=True, text=True,
                        env={k: v for k, v in os.environ.items() if k != "GH_TOKEN"},
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_text().splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
