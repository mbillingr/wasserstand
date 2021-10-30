from contextlib import contextmanager
from urllib.parse import urlparse


@contextmanager
def open_anywhere(uri, mode="r"):
    pr = urlparse(uri)

    if not pr.scheme:
        pass
    elif pr.scheme.lower() == "s3":
        import s3fs

        fs = s3fs.S3FileSystem()
        with fs.open(pr.netloc + pr.path, mode) as fd:
            yield fd
        return
    else:
        raise NotImplementedError(f"Can't open url scheme {pr.scheme}")

    with open(uri, mode) as fd:
        yield fd
