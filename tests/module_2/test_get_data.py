import sys
sys.path.append('/home/oscar/zrive-ds')

import unittest
from unittest.mock import patch, Mock
from src.module_2.get_data import *

class TestListObjectsInS3Directory(unittest.TestCase):

    @patch('src.module_2.get_data.boto3.Session')
    @patch('src.module_2.get_data.os.getenv')
    def test_list_objects_in_s3_directory(self, mock_getenv, mock_session):
        # Configurar los mocks
        fake_access_key = 'your_fake_value'
        fake_secret_key = 'your_fake_value'
        mock_getenv.side_effect = lambda key: fake_access_key if key == 'AWS_ACCESS_KEY_ID' else fake_secret_key

        mock_s3 = mock_session.return_value.client.return_value
        mock_s3.list_objects.return_value = {
            'Contents': [{'Key': 'file1.txt'}, {'Key': 'file2.txt'}]
        }

        # Llamar a la función que queremos probar
        bucket_name = 'your_bucket'
        directory_path = 'your_directory'
        list_objects_in_s3_directory(bucket_name, directory_path)

        # Asegurar que se llamó a los métodos y funciones esperados
        mock_getenv.assert_any_call('AWS_ACCESS_KEY_ID')
        mock_getenv.assert_any_call('AWS_SECRET_ACCESS_KEY')
        mock_session.assert_called_with(aws_access_key_id=fake_access_key, aws_secret_access_key=fake_secret_key)
        mock_s3.list_objects.assert_called_with(Bucket=bucket_name, Prefix=directory_path, Delimiter='/')


class TestDownloadDataFromS3(unittest.TestCase):

    @patch('src.module_2.get_data.boto3.Session')
    @patch('src.module_2.get_data.os.getenv')
    def test_download_data_from_s3(self, mock_getenv, mock_session):

        fake_access_key = 'your_fake_value'
        fake_secret_key = 'your_fake_value'
        mock_getenv.side_effect = lambda key: fake_access_key if key == 'AWS_ACCESS_KEY_ID' else fake_secret_key

        mock_s3 = mock_session.return_value.client.return_value

        bucket_name = 'your_bucket'
        bucket_path = 'your_bucket_path'
        save_path = 'your_save_path'
        download_data_from_s3(bucket_name, bucket_path, save_path)

        mock_getenv.assert_any_call('AWS_ACCESS_KEY_ID')
        mock_getenv.assert_any_call('AWS_SECRET_ACCESS_KEY')
        mock_session.assert_called_with(aws_access_key_id=fake_access_key, aws_secret_access_key=fake_secret_key)
        mock_s3.download_file.assert_called_with(bucket_name, bucket_path, save_path)


if __name__ == '__main__':
    unittest.main()