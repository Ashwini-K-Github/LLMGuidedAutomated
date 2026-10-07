#!/usr/bin/env python3
"""Runs the whole verification suite.

    python tests/run_all.py

Exits non-zero if any check fails, so it can be wired into CI. Sections needing
scikit-learn are skipped with a message when it is not installed.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.harness import Results, require


def main() -> int:
    results = Results()
    print("=" * 72)
    print("COCOON VERIFICATION SUITE")
    print("=" * 72)

    if require("pandas") is None:
        print("pandas is required to run this suite: pip install -r requirements.txt")
        return 1

    from tests import test_semantic_types, test_upload_robustness, test_student_workflows
    results.run("1. Semantic type inference (labelled ground truth)", test_semantic_types.run)
    results.run("2. Semantic type regression guards", test_semantic_types.run_regression_guards)
    results.run("3. Upload robustness and JSON safety", test_upload_robustness.run)
    results.run("3a. Reported student-data workflows", test_student_workflows.run)

    from tests import test_pipeline_quality as _layout
    results.run("3b. Frontend layout containment", _layout.run_layout_guards)

    if require("sklearn") is None:
        print("\n4-7. ML pipeline sections")
        print("-" * 24)
        results.skip("ML pipeline sections", "scikit-learn is not installed")
    else:
        from tests import test_pipeline_quality
        results.run("4. Problem type resolution", test_pipeline_quality.run_problem_type)
        results.run("4b. Feature column exclusion", test_pipeline_quality.run_column_exclusion)
        results.run("4c. Mutual information robustness", test_pipeline_quality.run_mi_robustness)
        results.run("4d. Feature selection reason clarity", test_pipeline_quality.run_feature_selection_reasons)
        results.run("5. Preprocessing fitted on the training split only", test_pipeline_quality.run_no_leakage)
        results.run("6. Feature selection relevance", test_pipeline_quality.run)
        results.run("7. Degenerate and tiny inputs", test_pipeline_quality.run_degenerate_inputs)

    return results.summary()


if __name__ == "__main__":
    sys.exit(main())
