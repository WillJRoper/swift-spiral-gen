# Repository Guidelines

## Project Structure & Module Organization
- `src/swift_spiral_ics/`: main package; CLI entrypoints live in `cli/` (`generate.py`, `visualize.py`, `movie.py`); physics logic in `physics/`; I/O helpers in `io/`; shared helpers in `utils/`.
- `tests/`: unit and integration coverage in `test_*.py`; mirrors package layout.
- `docs/theory.md`: reference for equations and assumptions—cite it when altering physics.
- `pyproject.toml`: dependencies, console scripts, `pytest` defaults, and `ruff` lint rules.

## Setup, Build, and Development Commands
- Python 3.9+; install with `pip install -e .[dev]` to get tests and linting tools.
- Generate ICs: `swift-spiral-ics examples/mw_m31_merger.yml` or another generator YAML config.
- Visualize: `swift-spiral-ics-viz demo.hdf5 --out-pdf diagnostics.pdf`; Movies: `swift-spiral-movie "snapshot_*.hdf5" --out-movie run.mp4`.
- Inspect options with `swift-spiral-ics --help`; prefer small box sizes in local smoke tests to keep runs fast.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; `ruff` enforces a 100-char line limit and import sorting (rule `I`).
- Use snake_case for modules/functions, CapWords for classes, and explicit units in docstrings/argument names.
- Keep randomness controlled via helpers in `utils/random.py`; pass seeds through generator YAML configs where feasible.
- Run `ruff check src tests` before committing; fix warnings or justify ignores locally.

## Testing Guidelines
- Add tests alongside affected code in `tests/`; name functions `test_*` and keep per-feature fixtures small.
- Core suite: `pytest`; target a quick loop with `pytest tests/test_physics.py -k orbits` when iterating.
- Coverage check: `pytest --cov=swift_spiral_ics --cov-report=term-missing`; flag gaps touching new logic.
- Do not commit outputs—`.gitignore` excludes `*.hdf5`, YAML, images, and movies; write tests to temp dirs.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subjects (e.g., `Add bar streaming option`); include rationale and any physics references in the body.
- PRs: explain the change, link issues, list commands used for validation, and note generated artifact locations (but avoid attaching large binaries).
- Request review on physics changes and highlight numerical assumptions or parameter defaults touched.
- Keep CLI help and README examples in sync when modifying flags or defaults.

## Operational & Safety Notes
- No secrets should be stored; configuration flows through CLI flags and generated YAML only.
- Large simulation outputs belong outside the repo (see `temp_frames/` and other ignored paths); document expected sizes when sharing results.
