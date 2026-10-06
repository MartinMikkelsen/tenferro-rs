import datetime
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.ci.runpod_cost import cost_report, parse_runpod_timestamp

ROOT = Path(__file__).resolve().parents[3]
UTC = datetime.timezone.utc


class ParseTimestampTests(unittest.TestCase):
    def test_runpod_rest_format(self) -> None:
        # Verbatim from the 2026-10-04 cleanup traceback (#2002).
        parsed = parse_runpod_timestamp("2026-10-04 11:09:24.633 +0000 UTC")
        self.assertEqual(parsed, datetime.datetime(2026, 10, 4, 11, 9, 24, 633000, UTC))

    def test_go_format_variants(self) -> None:
        cases = {
            "2026-10-04 11:09:24 +0000 UTC": datetime.datetime(2026, 10, 4, 11, 9, 24, tzinfo=UTC),
            "2026-10-04 11:09:24.123456789 +0000 UTC": datetime.datetime(2026, 10, 4, 11, 9, 24, 123456, UTC),
            "2026-10-04 20:09:24.5 +0900 JST": datetime.datetime(2026, 10, 4, 11, 9, 24, 500000, UTC),
            "2026-10-04 11:09:24.633 +0000": datetime.datetime(2026, 10, 4, 11, 9, 24, 633000, UTC),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_runpod_timestamp(text), expected)

    def test_iso_formats(self) -> None:
        expected = datetime.datetime(2026, 10, 4, 11, 9, 24, tzinfo=UTC)
        for text in ("2026-10-04T11:09:24Z", "2026-10-04T11:09:24+00:00", "2026-10-04T20:09:24+09:00"):
            with self.subTest(text=text):
                self.assertEqual(parse_runpod_timestamp(text), expected)

    def test_rejects_garbage_and_naive_times(self) -> None:
        for text in ("yesterday", "2026-10-04T11:09:24", ""):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_runpod_timestamp(text)


class CostReportTests(unittest.TestCase):
    NOW = datetime.datetime(2026, 10, 4, 11, 39, 24, 633000, UTC)

    def test_reports_paid_time_and_cost(self) -> None:
        lines, warnings = cost_report(
            {"adjustedCostPerHr": 0.44, "costPerHr": 0.40, "lastStartedAt": "2026-10-04 11:09:24.633 +0000 UTC"},
            self.NOW,
        )
        self.assertEqual(warnings, [])
        self.assertIn("RunPod paid time: 0.50h at $0.44/hr", lines)
        self.assertIn("RunPod estimated paid cost: $0.220", lines)

    def test_falls_back_to_list_price(self) -> None:
        lines, _ = cost_report(
            {"adjustedCostPerHr": None, "costPerHr": 0.40, "lastStartedAt": "2026-10-04T11:09:24.633Z"},
            self.NOW,
        )
        self.assertIn("RunPod paid time: 0.50h at $0.40/hr", lines)

    def test_missing_or_bad_fields_warn_instead_of_raising(self) -> None:
        for pod in (
            {},
            {"costPerHr": 0.4},
            {"costPerHr": 0.4, "lastStartedAt": "soon"},
            {"costPerHr": "0.4", "lastStartedAt": "2026-10-04T11:09:24Z"},
        ):
            with self.subTest(pod=pod):
                lines, warnings = cost_report(pod, self.NOW)
                self.assertEqual(lines, [])
                self.assertEqual(len(warnings), 1)

    def test_cli_prints_cost_and_never_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pod = Path(directory) / "pod.json"
            for payload, expect in (
                ({"costPerHr": 0.44, "lastStartedAt": "2026-10-04 11:09:24.633 +0000 UTC"}, "RunPod estimated paid cost: $"),
                ({"costPerHr": 0.44, "lastStartedAt": "garbage"}, "::warning::Unparseable RunPod lastStartedAt 'garbage'"),
                ("[]", "::warning::RunPod pod record unreadable"),
            ):
                with self.subTest(payload=payload):
                    pod.write_text(payload if isinstance(payload, str) else json.dumps(payload))
                    result = subprocess.run(
                        [sys.executable, "scripts/ci/runpod_cost.py", str(pod)],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(expect, result.stdout)


if __name__ == "__main__":
    unittest.main()
