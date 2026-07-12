"""Keep every test isolated from the user's real ~/.aim registry."""

import aim
import pytest


@pytest.fixture(autouse=True)
def isolate_aim_home(tmp_path, monkeypatch):
    monkeypatch.setattr(aim, "AIM_HOME", tmp_path / ".aim")
