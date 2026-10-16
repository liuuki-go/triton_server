"""CPU-only checks for report updates; no WSL, CUDA, or Triton calls."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("collect_environment", Path(__file__).with_name("collect_environment.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportFileTest(unittest.TestCase):
    def test_preserves_unknown_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system_env_info.txt"
            path.write_text("用户原有资料\n", encoding="utf-8")
            content = module.REPORT_MARKER + "\n新清单\n"
            backup = module.save_report(path, content)
            self.assertIsNotNone(backup)
            self.assertEqual(Path(backup).read_text(encoding="utf-8"), "用户原有资料\n")
            self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_generated_report_updates_without_accumulating_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "system_env_info.txt"
            module.save_report(path, module.REPORT_MARKER + "\nfirst\n")
            self.assertIsNone(module.save_report(path, module.REPORT_MARKER + "\nsecond\n"))
            self.assertEqual(path.read_text(encoding="utf-8"), module.REPORT_MARKER + "\nsecond\n")
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
