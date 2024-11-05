import requests
from datetime import datetime, timedelta
from functions.preprocess import get_point_geojson
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    SentinelHubCatalog,
    SHConfig,
)


def create_date_ranges(year):
    """
    Create a list of date ranges for each day in the given year.

    Parameters:
    - year (int): The year for which to create date ranges.

    Returns:
    - list: A list of dictionaries with 'from' and 'to' dates for each day in the year.
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = timedelta(days=1)

    date_ranges = []
    while start_date <= end_date:
        date_range = {
            "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
            "to": start_date.strftime("%Y-%m-%dT23:59:59Z"),
        }
        date_ranges.append(date_range)
        start_date += delta

    return date_ranges


def download_sentinel_image(
    config, coordinates, date, polygon_id, geometry_type="bbox"
):
    """
    Downloads a Sentinel-2 image using the Sentinel Hub API for a specified bounding box.

    Parameters:
    - config (SHConfig): A configuration object for the Sentinel Hub API.
    - coordinates (list): A list containing the bounding box coordinates in [minLon, minLat, maxLon, maxLat].
    - date (str): The date of the image in the format 'YYYY-MM-DD'.
    - polygon_id (int): The ID of the polygon.

    Returns:
    - None: The image is saved locally as 'output_image.tiff'.
    """

    # If the polygon ID folder deos not exist, create it
    if not os.path.exists(f"images/tiffs/{polygon_id}"):
        os.makedirs(f"images/tiffs/{polygon_id}")


    if geometry_type == "bbox":
        betsiboka_bbox = BBox(bbox=coordinates, crs=CRS.WGS84)
        betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=10)
    # Payload for the request
    # Evalscript for the request
    evalscript = """//VERSION=3
    function setup() {
      return {
        input: [{
          bands: [ "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",  "B11", "B12", "SCL"],
          units: "DN"
        }],
        output: {
          id: "default",
          bands: 11,
          sampleType: SampleType.UINT16
        }
      }
    }

    function evaluatePixel(sample) {
        return [  sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12 ];
    }"""

    request = SentinelHubRequest(
        data_folder=f"images/tiffs/{polygon_id}",
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date, date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=config,
    )
    request.get_data(save_data=True)
    # Headers for the request
    """
    # Fetch the image data
    print(len(true_color_imgs[0]))

    image = true_color_imgs[0]
    print(f"Image type: {image.dtype}")
    # Check the response

    # Create the directory if it does not exist
    print(f"Creating directory for polygon {polygon_id}")
    if not os.path.exists(f"dataset/images/{polygon_id}"):
        os.makedirs(f"dataset/images/{polygon_id}")

    with open(f"dataset/images/{polygon_id}/{date}.tiff", "wb") as file:
        file.write(image)
        """


def list_all_available_images(bbox, start_date, end_date, config):
    """
    List all available Sentinel-2 images using the Sentinel Hub Catalog API.

    Parameters:
    - bbox (list): A list containing the bounding box coordinates in [minLon, minLat, maxLon, maxLat].
    - start_date (str): The start date for the search in the format 'YYYY-MM-DD'.
    - end_date (str): The end date for the search in the format 'YYYY-MM-DD'.

    Returns:
    - list: A list of all available image metadata dictionaries.
    """

    # Configure Sentinel Hub

    # Define the bounding box
    bounding_box = BBox(bbox=bbox, crs=CRS.WGS84)

    # Define the time interval
    time_interval = (start_date, end_date)

    # Initialize the Catalog API
    bounding_box = BBox(bbox=bbox, crs=CRS.WGS84)

    # Define the time interval
    time_interval = (start_date, end_date)

    # Initialize the Catalog API
    catalog = SentinelHubCatalog(config=config)

    # Search for Sentinel-2 L2A data
    search_iterator = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,  # Corrected argument name
        bbox=bounding_box,
        time=time_interval,
        fields={"include": ["id", "geometry", "properties.datetime"]},
        limit=100,
    )

    # Collect all image metadata
    all_images = list(search_iterator)

    print(len(all_images))

    return all_images


def download_sentinel_images_in_range(
    config, coordinates, start_date, end_date, polygon_id, geometry_type="bbox"
):
    """
    Downloads multiple Sentinel-2 images using the Sentinel Hub API for a specified bounding box over a date range in one request.

    Parameters:
    - config (SHConfig): A configuration object for the Sentinel Hub API.
    - coordinates (list): A list containing the bounding box coordinates in [minLon, minLat, maxLon, maxLat].
    - start_date (str): The start date of the period in 'YYYY-MM-DD' format.
    - end_date (str): The end date of the period in 'YYYY-MM-DD' format.
    - polygon_id (int): The ID of the polygon.
    - geometry_type (str): The type of geometry ("bbox" by default).

    Returns:
    - None: Images are saved locally.
    """
    # Create directory for images if it does not exist
    if not os.path.exists(f"dataset/images/{polygon_id}"):
        os.makedirs(f"dataset/images/{polygon_id}")

    if geometry_type == "bbox":
        bbox = BBox(bbox=coordinates, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=10)

    # Evalscript for the request
    evalscript = """//VERSION=3
    function setup() {
      return {
        input: [{
          bands: [ "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",  "B11", "B12", "SCL"],
          units: "DN"
        }],
        output: {
          id: "default",
          bands: 10,
          sampleType: SampleType.UINT16
        }
      }
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12];
    }"""

    # Create a single request to download images for the specified time range
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),  # Correct parameter name
                mosaicking_order="mostRecent",  # Option to control how multiple images are handled
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    # Fetch the image data for all dates in the range
    images_data = request.get_data()

    print(images_data)

    # Save each image returned by the request
    for index, image in enumerate(images_data):
        date_str = start_date if len(images_data) == 1 else f"{start_date}_{index}"
        output_path = f"dataset/images/{polygon_id}/{date_str}.tiff"
        print(f"Saving image to: {output_path}")

        # Save the image
        with open(output_path, "wb") as file:
            file.write(image.tobytes())

    print(f"Completed downloading images from {start_date} to {end_date}.")
