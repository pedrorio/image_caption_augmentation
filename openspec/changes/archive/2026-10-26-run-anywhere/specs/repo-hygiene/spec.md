# repo-hygiene Specification

## Purpose

Keeps generated Python files out of git so clones do not carry bytecode and local caches stay untracked.

## ADDED Requirements

### Requirement: Bytecode is not tracked

The git index SHALL NOT contain compiled Python files (`*.pyc`).

#### Scenario: No tracked bytecode

- **WHEN** `git ls-files '*.pyc'` is run
- **THEN** it prints no paths

### Requirement: Python gitignore

The root `.gitignore` SHALL ignore `__pycache__/` directories and `*.pyc` files, plus a virtualenv directory named `.venv`. It SHALL NOT ignore `openspec/changes/archive`.

#### Scenario: Ignore rules present

- **WHEN** `.gitignore` is read
- **THEN** it contains `__pycache__/`, a pyc pattern, and `.venv/`

#### Scenario: OpenSpec archive is not ignored

- **WHEN** a file is added under `openspec/changes/archive/`
- **THEN** a root pattern named `archive` does not ignore that path
