# Maintainability Plan

Date: 2026-02-26
Owner: Thesis/Release workflow
Scope: Release scaffolding and housekeeping only

## Objectives
- Keep release artifacts organized and reproducible.
- Make validation repeatable with one script.
- Keep root directory clean by moving ad hoc notes into `notes/`.

## Repository Hygiene Baseline
- Canonical release folder: `thesis_release/<YYYY-MM-DD>/`.
- Canonical notes folder: `notes/`.
- Validation entrypoint: `scripts/validate_thesis_artifacts.py`.
- Ignore policy updated for transient release scratch outputs.

## Release Process (Minimal, Repeatable)
1. Create or update `thesis_release/<YYYY-MM-DD>/README.md`.
2. Move temporary or narrative report files from root into `notes/`.
3. Run `python scripts/validate_thesis_artifacts.py`.
4. Confirm no unintended file edits outside agreed scope before commit.

## Quality Gates
- Gate 1: Required files exist and are non-empty.
- Gate 2: Root-level transient note files are not present.
- Gate 3: `.gitignore` excludes transient release scratch outputs.
- Gate 4: Run `python scripts/check_large_modules.py` to track oversized modules before thesis freeze.

## Follow-Up Tasks
- Add checksum generation for future release bundles.
- Add CI job to run `scripts/validate_thesis_artifacts.py` on PRs touching release docs.
- Add a release manifest template under `thesis_release/templates/`.
