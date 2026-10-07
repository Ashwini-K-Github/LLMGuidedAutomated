"""Minimal assertion harness so the suite runs with no extra dependencies.

Usage:
    python tests/run_all.py
"""

import sys
import traceback


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def ok(self, name, detail=""):
        self.passed += 1
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.failed += 1
        self.failures.append((name, detail))
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))

    def skip(self, name, why):
        self.skipped += 1
        print(f"  SKIP  {name}  ({why})")

    def check(self, name, condition, detail=""):
        if condition:
            self.ok(name, detail)
        else:
            self.fail(name, detail)
        return bool(condition)

    def equals(self, name, actual, expected):
        return self.check(name, actual == expected, f"expected={expected!r} got={actual!r}")

    def run(self, name, fn):
        """Runs a section, catching exceptions so one failure cannot hide the rest."""
        print(f"\n{name}")
        print("-" * len(name))
        try:
            fn(self)
        except Exception:
            self.failed += 1
            self.failures.append((name, "raised an exception"))
            print(f"  ERROR in {name}:")
            traceback.print_exc()

    def summary(self) -> int:
        total = self.passed + self.failed
        print("\n" + "=" * 72)
        print(f"RESULT  {self.passed}/{total} checks passed"
              + (f", {self.skipped} skipped" if self.skipped else ""))
        if self.failures:
            print("\nFailures:")
            for name, detail in self.failures:
                print(f"  - {name}: {detail}")
        print("=" * 72)
        return 1 if self.failed else 0


def require(module_name):
    """Returns the module, or None when it is not installed."""
    try:
        return __import__(module_name)
    except ImportError:
        return None
