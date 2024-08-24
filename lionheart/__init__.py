from .utils.global_vars import INCLUDED_MODELS, REPO_URL


def get_version():
    import importlib.metadata

    return importlib.metadata.version("lionheart")


__version__ = get_version()
