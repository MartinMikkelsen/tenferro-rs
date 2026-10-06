import datetime
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.ci.runner_pin_check import PinCheckError, evaluate, read_pin

ROOT = Path(__file__).resolve().parents[3]
UTC = datetime.timezone.utc
SHA_335_1 = "4ef2f25285f0ae4477f1fe1e346db76d2f3ebf03824e2ddd1973a2819bf6c8cf"
SHA_337_0 = "70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613"


def release(version: str, published: str, sha: str | None = None, **extra) -> dict:
    body = "notes"
    if sha is not None:
        body += (
            f"\n- actions-runner-linux-x64-{version}.tar.gz "
            f"<!-- BEGIN SHA linux-x64 -->{sha}<!-- END SHA linux-x64 -->"
        )
    return {"tag_name": f"v{version}", "published_at": published, "body": body, **extra}


# actions/runner publication dates, as listed by the releases API.
HISTORY = [
    release("2.337.0", "2026-08-26T14:33:29Z", SHA_337_0),
    release("2.336.0", "2026-07-20T17:45:55Z", "1" * 64),
    release("2.335.1", "2026-06-09T01:32:05Z", SHA_335_1),
    release("2.335.0", "2026-06-08T17:08:55Z", "2" * 64),
]


def at(day: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(day).replace(tzinfo=UTC)


def visible(now: datetime.datetime) -> list[dict]:
    return [r for r in HISTORY if at(r["published_at"][:10]) <= now]


class ReadPinTests(unittest.TestCase):
    def test_reads_the_real_pin(self) -> None:
        text = (ROOT / ".github/workflows/runpod-gpu-execute.yml").read_text()
        version, sha = read_pin(text)
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertRegex(sha, r"^[0-9a-f]{64}$")

    def test_requires_exactly_one_pin(self) -> None:
        for text in ("", 'RUNNER_VERSION="1.2.3"\n', 'RUNNER_VERSION="1.2.3"\nRUNNER_VERSION="1.2.4"\nRUNNER_SHA256="' + "a" * 64 + '"\n'):
            with self.subTest(text=text):
                with self.assertRaises(PinCheckError):
                    read_pin(text)


class EvaluateTests(unittest.TestCase):
    def test_latest_pin_is_ok(self) -> None:
        verdict, _ = evaluate("2.337.0", SHA_337_0, HISTORY, at("2026-10-06"))
        self.assertEqual(verdict, "ok")

    def test_the_2026_09_outage_is_reported_before_it_happened(self) -> None:
        # 2.335.1 was rejected on 2026-09-24. With one newer release the
        # check warns, then fails after 14 days (2026-08-04), and fails at
        # once when the second newer release lands (2026-08-26).
        for day, expected in (
            ("2026-07-01", "ok"),
            ("2026-07-25", "warn"),
            ("2026-08-04", "fail"),
            ("2026-08-26", "fail"),
            ("2026-09-24", "fail"),
        ):
            with self.subTest(day=day):
                now = at(day)
                verdict, messages = evaluate("2.335.1", SHA_335_1, visible(now), now)
                self.assertEqual(verdict, expected, messages)

    def test_two_newer_releases_fail_immediately(self) -> None:
        verdict, messages = evaluate("2.335.1", SHA_335_1, HISTORY, at("2026-08-27"))
        self.assertEqual(verdict, "fail")
        self.assertIn("behind 2 newer release(s)", messages[-1])

    def test_checksum_mismatch_fails(self) -> None:
        verdict, messages = evaluate("2.337.0", "f" * 64, HISTORY, at("2026-10-06"))
        self.assertEqual(verdict, "fail")
        self.assertIn("does not match", messages[0])

    def test_unknown_or_prerelease_pin_fails(self) -> None:
        verdict, _ = evaluate("2.999.0", SHA_337_0, HISTORY, at("2026-10-06"))
        self.assertEqual(verdict, "fail")
        pre = [release("2.338.0", "2026-10-01T00:00:00Z", "3" * 64, prerelease=True)] + HISTORY
        verdict, _ = evaluate("2.338.0", "3" * 64, pre, at("2026-10-06"))
        self.assertEqual(verdict, "fail")

    def test_prereleases_and_drafts_do_not_count_as_newer(self) -> None:
        extra = [
            release("2.338.0", "2026-09-01T00:00:00Z", "3" * 64, prerelease=True),
            release("2.339.0", "2026-09-02T00:00:00Z", "4" * 64, draft=True),
        ]
        verdict, _ = evaluate("2.337.0", SHA_337_0, extra + HISTORY, at("2026-10-06"))
        self.assertEqual(verdict, "ok")

    def test_missing_checksum_in_notes_only_warns(self) -> None:
        history = [release("2.337.0", "2026-08-26T14:33:29Z")] + HISTORY[1:]
        verdict, messages = evaluate("2.337.0", SHA_337_0, history, at("2026-10-06"))
        self.assertEqual(verdict, "ok")
        self.assertTrue(messages[0].startswith("::warning::"))


class CliTests(unittest.TestCase):
    def run_cli(self, pin_version: str, pin_sha: str, now: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            pin = Path(directory) / "pin.yml"
            pin.write_text(f'  RUNNER_VERSION="{pin_version}"\n  RUNNER_SHA256="{pin_sha}"\n')
            releases = Path(directory) / "releases.json"
            releases.write_text(json.dumps(HISTORY))
            return subprocess.run(
                [sys.executable, "scripts/ci/runner_pin_check.py", "--pin-file", str(pin),
                 "--releases-json", str(releases), "--now", now],
                cwd=ROOT, capture_output=True, text=True,
            )

    def test_exit_status_follows_the_verdict(self) -> None:
        ok = self.run_cli("2.337.0", SHA_337_0, "2026-10-06T00:00:00Z")
        self.assertEqual(ok.returncode, 0, ok.stdout)
        self.assertIn("verdict=ok", ok.stdout)
        stale = self.run_cli("2.335.1", SHA_335_1, "2026-09-24T00:00:00Z")
        self.assertEqual(stale.returncode, 1, stale.stdout)
        self.assertIn("::error::Pinned runner 2.335.1 is behind 2 newer release(s)", stale.stdout)


if __name__ == "__main__":
    unittest.main()
