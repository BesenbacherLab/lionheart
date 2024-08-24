def get_version():
    import importlib.metadata

    return importlib.metadata.version("lionheart")


__version__ = get_version()
