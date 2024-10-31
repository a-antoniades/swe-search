from moatless.repository.file import CodeFile, FileRepository, UpdateResult

def __getattr__(name):
    if name == 'GitRepository':
        from moatless.repository.git import GitRepository
        return GitRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
