# Testing Guide

How we test things in ClipABit.

## Quick Start

```bash
cd backend
uv sync
uv run pytest
```


---

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Test individual components
│   ├── test_*.py
└── integration/             # Test multiple components together
    ├── test_*.py
```

**Unit tests** - One component at a time. Fast. No external dependencies.

**Integration tests** - Multiple components working together. May use mocks.

---

## Fixtures

All fixtures live in `conftest.py` and are automatically available to tests. Just add them as function parameters.

**Check conftest.py to see what's available.** Common ones:

```python
def test_something(chunker, sample_video_5s):
    """Fixtures handle all setup automatically."""
    chunks = chunker.chunk_video(str(sample_video_5s), "test")
    assert len(chunks) > 0
```

If you're repeating setup code, add a new fixture to conftest.py.

---

## Writing Tests

**Good test - specific, clear:**
```python
def test_chunks_respect_max_duration(sample_video_5s):
    """Verify chunks are not longer than max_duration."""
    chunker = Chunker(min_duration=1.0, max_duration=3.0)
    chunks = chunker.chunk_video(str(sample_video_5s), "test")

    for chunk in chunks:
        assert chunk.duration <= 3.0  # Clear expectation
```

**Bad test - vague, accepts anything:**
```python
def test_scene_threshold_affects_chunking(sample_video_5s):
    """Verify threshold does something."""
    chunker1 = Chunker(scene_threshold=5.0)
    chunker2 = Chunker(scene_threshold=20.0)

    chunks1 = chunker1.chunk_video(str(sample_video_5s), "test")
    chunks2 = chunker2.chunk_video(str(sample_video_5s), "test")

    # Doesn't actually verify different behavior
    assert len(chunks1) >= 1
    assert len(chunks2) >= 1
```

**Organization pattern:**
```python
class TestBasicFunctionality:
    """Test core features."""

    def test_creates_output(self, component):
        result = component.process(input_data)
        assert result is not None

class TestEdgeCases:
    """Test boundary conditions and error handling."""

    def test_empty_input_handles_gracefully(self, component):
        result = component.process([])
        assert result == []
```

**Naming:** `test_<what_it_does>_<condition>`
- Good: `test_chunker_respects_max_duration`
- Bad: `test_1`, `test_basic`, `test_works`

---

## Running Tests

```bash
# Basic
uv run pytest                    # All tests
uv run pytest -v                 # Verbose
uv run pytest -x                 # Stop on first fail
uv run pytest -s                 # Show print statements

# Specific tests
uv run pytest tests/unit/test_chunker.py
uv run pytest tests/unit/test_chunker.py::test_chunks_respect_max_duration
uv run pytest -k "duration"      # Match pattern

# Coverage
uv run pytest --cov              # Terminal report
uv run pytest --cov --cov-report=html  # HTML report (opens in browser)

# Performance
uv run pytest --durations=10     # Show slowest tests
uv run pytest -m "not slow"      # Skip slow tests
```

---

## Troubleshooting

**ModuleNotFoundError**
```bash
# Make sure you're in backend/
cd backend
uv run pytest
```

**Fixture not found**
- Check `conftest.py` for available fixtures

**Video generation fails**
```bash
# Make sure opencv-python (not headless) is installed for dev
uv sync --dev
```

---

## Conventions

**Do:**
- Write tests as you build features
- One test checks one thing
- Use descriptive names
- Use fixtures instead of repeating setup
- Test behavior, not implementation details

**Don't:**
- Make tests depend on each other
- Test private methods directly
- Use real external services (use mocks)
- Copy/paste setup code (make a fixture)
- Use random data without seed (makes tests flaky)

---

## Adding New Tests

1. **Decide where:** unit test (one component) or integration test (multiple components)?
2. **Create file:** `tests/unit/test_new_component.py` or `tests/integration/test_new_workflow.py`
3. **Write test:** Look at existing tests for examples
4. **Add fixtures if needed:** Add to `conftest.py` if you're repeating setup
5. **Run:** `uv run pytest tests/unit/test_new_component.py -v`

---

## Quick Reference

```bash
uv run pytest                    # All tests
uv run pytest -v                 # Verbose
uv run pytest -x                 # Stop on first fail
uv run pytest --cov              # With coverage
uv run pytest tests/unit/        # Just unit tests
```

For more examples, check the existing tests and conftest.py.
