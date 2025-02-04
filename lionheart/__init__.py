from .utils.global_vars import INCLUDED_MODELS, REPO_URL


def get_version():
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata

    return metadata.version("lionheart")


__version__ = get_version()
