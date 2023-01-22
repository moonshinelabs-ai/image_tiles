import os
from multiprocessing.connection import Client

try:
    from botocore.exceptions import ClientError  # pants: no-infer-dep
except ImportError:
    pass

from smart_open import open

from .glob import glob


def _aws_exists(path: str) -> bool:
    """Test if an AWS file exists."""
    file_exists = False

    # Check for individual files.
    try:
        with open(path):
            file_exists = True
    except OSError:
        pass

    # Check for "folder", which means that folder has items
    # since S3 cannot have empty folders.
    try:
        items = glob(os.path.join(path, "*"))
        if items:
            file_exists = True
    except ClientError:
        pass

    return file_exists


def _gcs_exists(path: str) -> bool:
    """Test if a GCS file exists."""
    raise NotImplementedError


def _local_exists(path: str) -> bool:
    """Test if a local file exists."""
    return os.path.exists(path)


def exists(path: str) -> bool:
    """Check for existance of a file, maybe in the cloud.

    Args:
        path: An S3 path including the s3://

    Returns:
        exists: File exists on S3 at path given.
    """
    if "s3://" in path:
        return _aws_exists(path)
    elif "gs://" in path:
        return _gcs_exists(path)
    else:
        return _local_exists(path)
