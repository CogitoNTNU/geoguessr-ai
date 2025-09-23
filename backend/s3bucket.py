import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import boto3

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
aws_region = os.getenv("AWS_REGION")


def convert_image_to_binary(image_path):
    # Open the image and convert to binary

    with Image.open(image_path) as img:
        img_byte_array = BytesIO()
        img.save(img_byte_array, format=img.format)
        img_byte_array.seek(0)

    return img_byte_array


def upload_image_to_s3(image_path, s3_key):
    # Initialize the S3 client with credentials

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )

    # Convert image to binary data
    image_data = convert_image_to_binary(image_path)

    # Upload to S3
    try:
        s3_client.upload_fileobj(image_data, aws_bucket_name, s3_key)
        print(f"Image successfully uploaded to S3 at {aws_bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading image: {e}")
