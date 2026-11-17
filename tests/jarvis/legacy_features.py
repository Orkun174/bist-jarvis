"""The original feature functions, for the parity tests in test_features.py."""

from __future__ import annotations

from tests.jarvis import legacy_reference


def legacy_functions() -> dict:
    return vars(legacy_reference)
