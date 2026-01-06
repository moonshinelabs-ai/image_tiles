import unittest
from unittest import mock

import boto3
import moto
from parameterized import parameterized

from .file_exists import _aws_exists, _gcs_exists, exists


class Test(unittest.TestCase):
    @parameterized.expand(
        [
            ("s3://path/cat.png", True),
            ("s3://path/koala.png", False),
            ("s3://newpath/cat.png", False),
        ]
    )
    def test_aws_exists(self, path, result):
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

            exist = _aws_exists(path)
            self.assertEqual(exist, result)

            exist = exists(path)
            self.assertEqual(exist, result)


class MockBlob:
    """Mock GCS blob object."""

    def __init__(self, name: str):
        self.name = name


def _setup_gcs_mock(mock_blobs):
    """Set up GCS mock modules and return the modules dict."""
    mock_storage = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_bucket = mock.MagicMock()
    mock_bucket.list_blobs.return_value = mock_blobs
    mock_client.bucket.return_value = mock_bucket
    mock_storage.Client.return_value = mock_client

    mock_google_cloud = mock.MagicMock()
    mock_google_cloud.storage = mock_storage

    mock_exceptions = mock.MagicMock()
    mock_google_cloud.exceptions = mock_exceptions

    return {
        "google": mock.MagicMock(),
        "google.cloud": mock_google_cloud,
        "google.cloud.storage": mock_storage,
        "google.cloud.exceptions": mock_exceptions,
    }


class TestGcsExists(unittest.TestCase):
    @parameterized.expand(
        [
            ("gs://path/cat.png", True),
            ("gs://path/koala.png", False),
            ("gs://newpath/cat.png", False),
        ]
    )
    def test_gcs_exists(self, path, result):
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
            with mock.patch("image_tiles.file_exists.open") as mock_open:
                # Configure mock_open to raise OSError for non-existent files
                def open_side_effect(p):
                    if p in files:
                        return mock.MagicMock()
                    raise OSError(f"File not found: {p}")

                mock_open.side_effect = open_side_effect

                exist = _gcs_exists(path)
                self.assertEqual(exist, result)

                exist = exists(path)
                self.assertEqual(exist, result)


if __name__ == "__main__":
    unittest.main()
