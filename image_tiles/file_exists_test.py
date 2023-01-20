import unittest

import boto3
import moto
from parameterized import parameterized

from .file_exists import _aws_exists, exists


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


if __name__ == "__main__":
    unittest.main()
