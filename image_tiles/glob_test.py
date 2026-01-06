import unittest
from typing import Sequence
from unittest import mock

import boto3
import moto
from parameterized import parameterized

from .glob import _aws_glob, _filter_sequences, _gcs_glob


def index_from_lists(a: Sequence, b: Sequence) -> Sequence:
    """Return a list of indexes if an element of a is in b."""
    indexes = []
    for idx, item in enumerate(b):
        if item in a:
            indexes.append(idx)

    return indexes


class TestGlob(unittest.TestCase):
    @parameterized.expand(
        [
            ("path/to/file/*.png", [0, 1, 2, 3]),
            ("path/to/file/*.jpg", [4, 5]),
            ("path/to/file/*.txt", [6]),
            ("path/to/file/cat*.png", [1, 3]),
            ("path/to/file/cat*", [1, 3, 5, 6]),
        ]
    )
    def test_filter_sequences(self, pattern, result_indexes):
        possible_results = (
            "path/to/file/001.png",
            "path/to/file/cat.png",
            "path/to/file/dog.png",
            "path/to/file/catdog.png",
            "path/to/file/dog.jpg",
            "path/to/file/catdog.jpg",
            "path/to/file/catdog.txt",
        )

        results = _filter_sequences(pattern, possible_results)
        indexes = index_from_lists(results, possible_results)
        self.assertEqual(indexes, result_indexes)

    @moto.mock_s3
    def test_s3_empty_glob(self):
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="path")

        # Test an empty bucket
        results = _aws_glob("s3://path/*.png")
        self.assertEqual(results, [])

    @parameterized.expand(
        [
            ("s3://path/*.png", [0, 1, 2, 3]),
            ("s3://path/*.jpg", [4, 5]),
            ("s3://path/*.txt", [6]),
            ("s3://path/cat*.png", [1, 3]),
            ("s3://path/cat*", [1, 3, 5, 6]),
        ]
    )
    def test_s3_glob(self, pattern, result_indexes):
        with moto.mock_s3():
            conn = boto3.resource("s3", region_name="us-east-1")
            bucket_name = "path"
            conn.create_bucket(Bucket=bucket_name)

            # Add a bunch of files
            files = (
                "s3://path/001.png",
                "s3://path/cat.png",
                "s3://path/dog.png",
                "s3://path/catdog.png",
                "s3://path/dog.jpg",
                "s3://path/catdog.jpg",
                "s3://path/catdog.txt",
            )
            s3 = boto3.client("s3", region_name="us-east-1")
            for f in files:
                filename = f.split("/")[-1]
                s3.put_object(Bucket=bucket_name, Key=filename, Body="nodata")

            results = _aws_glob(pattern)
            indexes = index_from_lists(results, files)
            self.assertEqual(indexes, result_indexes)


class MockBlob:
    """Mock GCS blob object."""

    def __init__(self, name: str):
        self.name = name


def _setup_gcs_mock(mock_blobs):
    """Set up GCS mock modules and return the storage mock."""
    mock_storage = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_bucket = mock.MagicMock()
    mock_bucket.list_blobs.return_value = mock_blobs
    mock_client.bucket.return_value = mock_bucket
    mock_storage.Client.return_value = mock_client

    mock_google_cloud = mock.MagicMock()
    mock_google_cloud.storage = mock_storage

    return {
        "google": mock.MagicMock(),
        "google.cloud": mock_google_cloud,
        "google.cloud.storage": mock_storage,
    }


class TestGcsGlob(unittest.TestCase):
    def test_gcs_empty_glob(self):
        modules = _setup_gcs_mock([])
        with mock.patch.dict("sys.modules", modules):
            results = _gcs_glob("gs://path/*.png")
            self.assertEqual(results, [])

    @parameterized.expand(
        [
            ("gs://path/*.png", [0, 1, 2, 3]),
            ("gs://path/*.jpg", [4, 5]),
            ("gs://path/*.txt", [6]),
            ("gs://path/cat*.png", [1, 3]),
            ("gs://path/cat*", [1, 3, 5, 6]),
        ]
    )
    def test_gcs_glob(self, pattern, result_indexes):
        files = (
            "gs://path/001.png",
            "gs://path/cat.png",
            "gs://path/dog.png",
            "gs://path/catdog.png",
            "gs://path/dog.jpg",
            "gs://path/catdog.jpg",
            "gs://path/catdog.txt",
        )

        mock_blobs = [MockBlob(f.replace("gs://path/", "")) for f in files]
        modules = _setup_gcs_mock(mock_blobs)

        with mock.patch.dict("sys.modules", modules):
            results = _gcs_glob(pattern)
            indexes = index_from_lists(results, files)
            self.assertEqual(indexes, result_indexes)


if __name__ == "__main__":
    unittest.main()
