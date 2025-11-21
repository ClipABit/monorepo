# Testing Guide

Quick guide on how we test things in this project.

## Setup

First time:
```bash
cd backend
uv sync
```

Run tests:
```bash
uv run pytest
```

---

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures for all tests
├── unit/                    # Test individual components in isolation
│   └── test_*.py
└── integration/             # Test multiple components working together
    └── test_*.py
```

**Unit tests** - One component at a time. Fast. No external dependencies.

**Integration tests** - Multiple components. Slower. May use mocks for external services.

---

## Writing Tests

**Pick where it goes:**
- Testing one class/function in isolation? → `tests/unit/`
- Testing how multiple things work together? → `tests/integration/`

**Basic pattern:**
```python
def test_something_specific(fixture_name):
    """Short description of what this checks."""
    # Arrange - fixtures handle setup automatically

    # Act
    result = do_the_thing()

    # Assert
    assert result == expected
```

**Naming convention:**
- File: `test_component_name.py`
- Function: `test_what_it_does`
- Class (optional): `class TestComponentName:`

---

## Fixtures

Fixtures live in `conftest.py` and are automatically available to all tests.

**What they are:** Reusable test setup/data so you don't repeat yourself.

**How to use them:** Just add as function parameters.

```python
# Instead of this:
def test_something():
    video_path = generate_test_video()
    component = Component(config)
    result = component.process(video_path)

# Do this:
def test_something(component, sample_video_5s):
    result = component.process(str(sample_video_5s))
```

**Common fixture types:**
- Component instances (pre-configured with test settings)
- Test data (videos, frames, arrays)
- Mocks (fake external services like Modal Dict)

Check `conftest.py` to see what's available. Add new fixtures there when you find yourself repeating setup code.

---

## Running Tests

```bash
# Everything
uv run pytest

# Specific file
uv run pytest tests/unit/test_something.py

# Specific test
uv run pytest tests/unit/test_something.py::test_specific_thing

# Pattern matching
uv run pytest -k "pattern"

# Verbose output
uv run pytest -v

# Show print statements
uv run pytest -s

# Stop on first failure
uv run pytest -x

# Coverage report
uv run pytest --cov
uv run pytest --cov --cov-report=html  # HTML version
```

---

## Conventions

**Do:**
- Write tests as you build features
- One test checks one thing
- Use descriptive test names (`test_chunker_respects_max_duration`)
- Use fixtures instead of repeating setup
- Test behavior, not implementation details

**Don't:**
- Make tests depend on each other
- Test private methods directly
- Use real external services (use mocks)
- Copy/paste setup code (make a fixture)

---

## Mocking External Services

For things like Modal Dict, Pinecone, S3 - use mocks.

```python
def test_with_modal_dict(mock_modal_dict):
    # mock_modal_dict acts like a regular dict
    # No real Modal infrastructure needed
    connector = JobStoreConnector("test")
    connector.create_job("123", {"status": "processing"})
    assert "123" in mock_modal_dict
```

Add new mocks to `conftest.py` when you need to fake external services.

---

## Troubleshooting

**Module not found:**
Make sure you're in the `backend/` directory.

---

## Quick Reference

```bash
uv run pytest                    # Run all tests
uv run pytest -v                 # Verbose
uv run pytest -x                 # Stop on first fail
uv run pytest -s                 # Show prints
uv run pytest -k "pattern"       # Match pattern
uv run pytest --cov              # Coverage report
uv run pytest tests/unit/        # Just unit tests
```

Check existing tests for examples.
