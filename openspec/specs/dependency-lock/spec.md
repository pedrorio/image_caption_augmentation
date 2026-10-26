# dependency-lock Specification

## Purpose
Pins the Python dependencies for training and evaluation so a new environment installs the same packages, including scikit-learn under its current name.

## Requirements

### Requirement: Lock file

The repository SHALL pin dependencies in `uv.lock`, generated from `pyproject.toml`. A frozen install from that lock SHALL be the supported way to recreate the environment. CPU PyTorch wheels SHALL be the pinned torch build.

#### Scenario: Lock file exists

- **WHEN** the repository root is listed
- **THEN** `pyproject.toml` and `uv.lock` are both present

#### Scenario: Frozen install

- **WHEN** `uv sync --frozen --all-groups` is run on a supported Python
- **THEN** the command installs the locked set without resolving a new torch build

### Requirement: scikit-learn name

The project SHALL depend on `scikit-learn` and SHALL NOT depend on the PyPI distribution named `sklearn`.

#### Scenario: Project metadata

- **WHEN** `pyproject.toml` is read
- **THEN** it lists `scikit-learn` and does not list a `sklearn` dependency
