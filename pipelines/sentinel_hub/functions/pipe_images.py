import csv
from functions.preprocess import calculate_bounding_box_area
from functions.sentinel_hub_request import (
    list_all_available_images,
    download_sentinel_image,
)
import pandas as pd
from dotenv import load_dotenv
import os
from sentinelhub import SHConfig

load_dotenv(dotenv_path="env/.env.local")
# Define the path to the .env.local file
env_path = os.path.join("env", ".env.local")
if os.path.exists("env/.env.local"):
    print(f"Loading environment from {env_path}")
else:
    print("File not found:", env_path)


# Create a configuration object
config = SHConfig()

# Set your client ID, client secret, and instance ID
config.sh_client_id = os.getenv("CLIENT_ID")
config.sh_client_secret = os.getenv("CLIENT_SECRET")

config.save()


import csv
import os

def register_image(images, output_file="images.csv"):
    """
    Registers the image data to a CSV file.

    Parameters:
    - images (list): A list of image metadata dictionaries from the Sentinel Hub Catalog API.
    - output_file (str): The name of the output CSV file. Defaults to 'images.csv'.

    Returns:
    - None
    """
    # Check if the CSV file already exists to decide whether to write a header
    file_exists = os.path.isfile(output_file)
    
    # Open the file in append mode if it exists, write mode if it doesn't
    with open(output_file, "a" if file_exists else "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row if the file does not already exist
        if not file_exists:
            writer.writerow(["Image ID", "Date", "Cloud Cover"])

        # Write each image's data
        for image in images:
            # Get the image ID, date, and cloud cover
            image_id = image["id"]
            image_date = image["properties"]["datetime"]
            cloud_cover = image["properties"].get("eo:cloud_cover", "N/A")

            # Write the data row
            writer.writerow([image_id, image_date, cloud_cover])

    print(f"Image data has been {'appended to' if file_exists else 'written to'} '{output_file}'.")



def process_sentinel_images(
    point, point_id, area_hectares=50, start_date="2022-01-01", end_date="2022-12-31"
):
    """
    Calculate the bounding box for a given point, list available Sentinel-2 images for a specified date range,
    save the image metadata to a CSV file, and download the images.

    Parameters:
    - point (tuple): Latitude and Longitude of the point (lat, lon).
    - point_id (str): Identifier for the point used in naming output files.
    - area_hectares (int, optional): Size of the bounding box area in hectares. Default is 5 hectares.
    - start_date (str, optional): Start date for searching images, in 'YYYY-MM-DD' format. Default is '2022-01-01'.
    - end_date (str, optional): End date for searching images, in 'YYYY-MM-DD' format. Default is '2022-12-31'.
    """
    # Initialize Sentinel Hub configuration
    config = SHConfig()

    # Calculate the bounding box
    bbox = calculate_bounding_box_area(point, area_hectares)

    # List available images using the Sentinel Hub Catalog API
    images = list_all_available_images(bbox, start_date, end_date, config)

    # Register the image metadata to a CSV file
    output_file = f"images/image_list/{point_id}_images.csv"
    register_image(images, output_file=output_file)

    print(f"Found {len(images)} images for the year {start_date[:4]}.")

    # Load the CSV file containing the image metadata
    df = pd.read_csv(output_file)

    # Download images based on the metadata
    for _, row in df.iterrows():
        print(f"Downloading image for date: {row['Date']}")
        download_sentinel_image(config, bbox, row["Date"], point_id)
