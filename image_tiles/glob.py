import fnmatch
import glob as sys_glob
import os
from typing import Sequence

from loguru import logger


def _filter_sequences(pattern: str, search_list: Sequence) -> list:
    """Return only the items from search_list that match pattern."""

    def match_file(filename):
        return fnmatch.fnmatch(filename, pattern)

    return list(filter(match_file, search_list))


def _aws_glob(path: str) -> list:
    """Implementation of AWS glob."""
    try:
        import boto3
    except ImportError:
        logger.error(
            "Didn't find AWS dependencies, try installing image_tiles[aws] if you haven't already."
        )
        return []

    # Do a bunch of surgery on the path to get the components
    full_path, _ = os.path.split(path)
    _, _, bucket, *remaining = full_path.split("/")
    prefix = "/".join(remaining)

    # For wildcard folder globs (i.e. /path/to/**/*.jpg) we need to do a
    # bit more processing.
    prefix = prefix.replace("/**", "")

    # List all the objects in the path via pagination since there could
    # be more than 1000 objects in our bucket
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    filenames = []
    for page in pages:
        # Construct the filenames from the results
        if "Contents" in page:
            keys = [f"s3://{bucket}/{o['Key']}" for o in page["Contents"]]

            # Add to our results
            filenames.extend(keys)

    # Do glob filtering
    return _filter_sequences(path, filenames)


def _gcs_glob(path: str) -> list:
    """Implementation of GCS glob."""
    raise NotImplementedError


def _local_glob(path: str) -> list:
    """Implementation of local glob."""
    return sys_glob.glob(path)


def glob(path: str) -> list:
    """Glob with support for cloud services.

    Args:
        path: An input path, includes the uri like s3:// or gs:// if in the cloud.

    Returns:
        results: A list of files returned from the glob.
    """
    if "s3://" in path:
        return _aws_glob(path)
    elif "gs://" in path:
        return _gcs_glob(path)
    else:
        return _local_glob(path)
