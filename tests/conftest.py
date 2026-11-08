"""Shared pytest fixtures.

The license guard (Issue #13) treats destinations outside the project
workspace as public distribution locations. Existing tests legitimately
write MentalChat16K-derived files into pytest's ``tmp_path`` to exercise
local processing semantics — so each test's ``tmp_path`` is anchored as the
project workspace (``config.ROOT``), making those writes private/local.

Tests that specifically exercise the public/private boundary (see
``tests/test_license_guard.py``) re-anchor ``config.ROOT`` themselves and
use destinations outside both the anchored workspace and the real managed
roots. Production behavior is unchanged: the real ``config.ROOT`` (with
``data/``/``outputs/``/caches) remains the private zone.
"""
from __future__ import annotations

import pytest

import config


@pytest.fixture(autouse=True)
def _license_guard_workspace(tmp_path, monkeypatch):
    """Treat each test's tmp_path as the private project workspace."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
