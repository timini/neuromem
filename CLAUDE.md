# neuromem Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-11

## Active Technologies

- Python 3.10+ (uses PEP 604 `X | None` unions, `typing.Literal` for status discriminators, `zip(..., strict=True)`). + Python 3.10+ standard library; **numpy** (>= 1.26) and **pandas** (>= 2.1) as first-class runtime dependencies per Constitution v2.0.0 Principle II. No other runtime dependencies in `neuromem-core`. Dev dependencies (via `[dependency-groups] dev`): `pytest`, `pytest-cov`, `ruff`, `pre-commit`. (001-neuromem-core)

## Project Structure

```text
src/
tests/
```

## Commands

cd src && pytest && ruff check .

## Code Style

Python 3.10+ (uses PEP 604 `X | None` unions, `typing.Literal` for status discriminators, `zip(..., strict=True)`).: Follow standard conventions

## Recent Changes

- 001-neuromem-core: Added Python 3.10+ (uses PEP 604 `X | None` unions, `typing.Literal` for status discriminators, `zip(..., strict=True)`). + Python 3.10+ standard library; **numpy** (>= 1.26) and **pandas** (>= 2.1) as first-class runtime dependencies per Constitution v2.0.0 Principle II. No other runtime dependencies in `neuromem-core`. Dev dependencies (via `[dependency-groups] dev`): `pytest`, `pytest-cov`, `ruff`, `pre-commit`.

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
