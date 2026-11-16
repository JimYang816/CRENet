"""Place pytest's temporary files alongside other smoke-test artifacts."""

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if config.getoption("basetemp") is None:
        temp_root = Path(config.rootpath) / "outputs" / "smoke" / "pytest"
        temp_root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = temp_root / uuid4().hex
