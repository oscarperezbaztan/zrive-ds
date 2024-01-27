import boto3
import os
from botocore.exceptions import NoCredentialsError


def list_objects_in_s3_directory(bucket_name, directory_path):
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        s3 = session.client('s3')

        response = s3.list_objects(Bucket=bucket_name, Prefix=directory_path, Delimiter='/')

        for obj in response.get('Contents', []):
            print(obj['Key'])

    except NoCredentialsError:
        print("Credenciales no disponibles. Asegúrate de configurar las variables de entorno AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY.")


def download_data_from_s3(bucket_name, bucket_path, save_path):
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        s3 = session.client('s3')

        s3.download_file(bucket_name, bucket_path, save_path)

        print(f"Archivo descargado y guardado en: {save_path}")

    except NoCredentialsError:
        print("Credenciales no disponibles. Asegúrate de configurar las variables de entorno AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY.")


#list_objects_in_s3_directory('zrive-ds-data', 'groceries/box_builder_dataset/')
#download_data_from_s3('zrive-ds-data', 'groceries/sampled-datasets/regulars.parquet', '/home/oscar/data/regulars.parquet')
#download_data_from_s3('zrive-ds-data', 'groceries/sampled-datasets/abandoned_carts.parquet', '/home/oscar/data/abandoned_carts.parquet')
#download_data_from_s3('zrive-ds-data', 'groceries/sampled-datasets/inventory.parquet', '/home/oscar/data/inventory.parquet')
#download_data_from_s3('zrive-ds-data', 'groceries/sampled-datasets/users.parquet', '/home/oscar/data/users.parquet')
#download_data_from_s3('zrive-ds-data', 'groceries/box_builder_dataset/feature_frame.csv', '/home/oscar/data/feature_frame.csv')
