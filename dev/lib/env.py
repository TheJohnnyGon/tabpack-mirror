import getpass
import tomllib
from pathlib import Path
from typing import Any

import lib.env

_SETTINGS_FILE_NAME = 'settings.toml'


def get_dev_dir() -> Path:
    return lib.env.get_project_dir() / 'dev'


def get_user_dir() -> Path:
    return get_dev_dir() / 'users' / getpass.getuser()


def get_dev_settings_path() -> Path:
    return get_dev_dir() / _SETTINGS_FILE_NAME


def get_user_settings_path() -> Path:
    return get_user_dir() / _SETTINGS_FILE_NAME


def _load_settings(path: Path) -> dict[str, Any]:
    with open(path, 'rb') as f:
        return tomllib.load(f)


def load_dev_settings() -> dict[str, Any]:
    return _load_settings(get_dev_settings_path())


def load_user_settings() -> dict[str, Any]:
    return _load_settings(get_user_settings_path())
