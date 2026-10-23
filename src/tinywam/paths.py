"""Repository defaults with explicit overrides for external data and dependencies."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def configured_path(variable, default, relative=''):
    root = Path(os.environ.get(variable, str(default))).expanduser().resolve()
    return str(root / relative)


def repo_path(relative=''):
    return str(ROOT / relative)


def data_path(relative=''):
    return configured_path('TINYWAM_DATA_ROOT', ROOT / 'data', relative)


def model_path(relative=''):
    relative = relative.replace('flow_tokenizers_v5', 'tokenizers', 1)
    return configured_path('TINYWAM_MODEL_ROOT', ROOT / 'artifacts', relative)


def external_path(component, relative=''):
    variable = 'TINYWAM_' + component.upper().replace('-', '_') + '_ROOT'
    return configured_path(variable, ROOT / 'external' / component, relative)


def cache_path(relative=''):
    default = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'tinywam'
    return configured_path('TINYWAM_CACHE_ROOT', default, relative)


def asset_filename(filename):
    """Resolve saved URDF asset references without knowing their original machine."""
    root = Path(configured_path('TINYWAM_ASSETS_ROOT', ROOT / 'assets'))
    parts = filename.replace('\\', '/').split('/assets/', 1)
    if len(parts) == 2:
        return str(root / parts[1])
    return filename
