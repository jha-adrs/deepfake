import boto3
from botocore.exceptions import NoCredentialsError

class S3Util:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name='us-east-1'):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.bucket_name = bucket_name

    def upload_file(self, file_path, s3_key):
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            return s3_key
        except FileNotFoundError:
            print("The file was not found")
            return None
        except NoCredentialsError:
            print("Credentials not available")
            return None

    def get_file(self, s3_key, download_path):
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, download_path)
            return download_path
        except FileNotFoundError:
            print("The file was not found")
            return None
        except NoCredentialsError:
            print("Credentials not available")
            return None

# Example usage:
s3_util = S3Util('access', 'secret', 'assets')
s3_key = s3_util.upload_file('path', 'desired_s3_key')
download_path = s3_util.get_file('s3_key', 'path/to/save/file')