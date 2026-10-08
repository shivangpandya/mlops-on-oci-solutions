# ReviewOps local milestone: implementation and review notes

## Decisions

- Work in the approved folder without Git worktrees or commits: this folder is not a Git repository. Source hashes and retained reports provide provenance, but Git history/rollback remains unavailable until a repository is initialized.
- Acquire pinned IMDb Parquet using Hugging Face Hub and pandas/PyArrow directly, rather than introducing the datasets library. This reduces dependencies; an upstream path change will fail clearly and require an acquisition update.
- Use numeric registry versions and an explicit model signature. Installed MLflow entity versions are normalized to integers; prediction token counts are non-null integers. Incompatible signature outputs should fail validation.
- Keep successful evaluation status separate from a failed quality gate. A run can finish evaluation successfully yet reject a release; startup, warm-up, and data-preparation failures receive MLflow FAILED status.
- Registry/tag/alias errors return failure with no successful release record. They can leave a numbered registry entry if a later operation fails; it must not be treated as a released candidate without checking alias/report state.
- Trust the local registry and checkpoint provenance. Hash checks detect changed files; they are not signatures against a malicious actor replacing manifests, files, or model code. Only load trusted model packages.
- Preserve historical evaluator hashes. New code produces new evaluation runs; historical metrics are not rewritten to pretend they used later code.

## Independent review and fixes

A fresh reviewer checked the local milestone. The main finding was startup/warm-up failures escaping before any failed evaluation evidence could be recorded. Regression tests reproduced the issue, then shared orchestration was changed to save failed reports, record failed runs when tracking is available, return nonzero, and avoid registration. A preliminary local report also replaces stale successful evidence before connecting to the tracking store.

Review also identified outdated beginner status text, missing dependency-file hashes, and fresh-install test instructions that omitted MLflow installation. These affect presentation accuracy, planned reproducibility, and a first-time learner's ability to verify the project, so they were corrected in the same final pass.

Tests use real checkpoint inference and temporary local SQLite registries. Failure injection is limited to conditions that cannot be reproduced reliably without intentionally breaking dependencies, including unavailable registration and non-finite/startup failures. The final suite result and live checks are retained in the execution ledger.

## Review boundaries

- Containers, OCI deployment, KServe, DVC, and Kubeflow are later stages. Linux image compatibility, resource sizing, and cloud behavior still need verification.
- The reviewer did not perform clean package installation or public-data reacquisition. The implementer installed pinned direct dependencies in the project environment, downloaded the actual public data, ran it, and verified `pip check`; this is not a clean Linux build test.
- The reviewer did not rerun mutation-producing integration tests or offline artifact reload. The implementer ran those checks, including registry inference from outside the project directory with upstream access disabled.
- Concurrent artifact mutation and malicious registry code were outside this trusted local learning setup. Production trust and immutable artifact distribution need a separate design.

No deferred code-polish findings remain from this review. No cloud resources were created or modified.
