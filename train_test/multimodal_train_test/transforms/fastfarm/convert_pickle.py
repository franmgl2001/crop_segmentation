import os
import json
import rasterio
import rasterio.features
import rasterio.mask
import rasterio.warp
import pandas as pd
from shapely import wkt
from shapely.geometry import mapping, Polygon
import numpy as np
import pickle


def get_folders_with_files(directory):
    folders = []

    # Walk through the main directory
    for root, _, files in os.walk(directory):
        # Check if there are .json or .tiff files in the folder
        has_json_or_tiff = any(
            file.endswith(".json") or file.endswith(".tiff") for file in files
        )

        if has_json_or_tiff:
            folder_name = os.path.basename(root)
            folders.append(folder_name)

    return folders


def extract_time_from(folder_name, root="../tiffs"):
    # Construct the path to the JSON file
    json_file_path = os.path.join(root, folder_name, "request.json")

    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"{json_file_path} does not exist.")
        return None

    # Open and load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

        try:
            # Extract the 'from' value in the 'timeRange'
            time_from = data["request"]["payload"]["input"]["data"][0]["dataFilter"][
                "timeRange"
            ]["from"]
            return time_from
        except KeyError:
            print(f"'from' key not found in {json_file_path}")
            return None


def transform_mask(path, polygon):
    with rasterio.open(path) as src:
        # Transform the polygon to the same CRS as the image
        polygon = rasterio.warp.transform_geom(
            {"init": "EPSG:4326"}, src.crs, mapping(polygon)
        )

        mask = np.zeros((src.height, src.width), dtype=np.uint8)
        # Rasterize the polygon into the mask
        mask = rasterio.features.geometry_mask(
            [polygon],
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width),
            all_touched=True,
        )

        # Calculate the area with the highest density of masked pixels
        kernel_size = 24  # Adjust this to the desired crop size
        density = np.lib.stride_tricks.sliding_window_view(
            mask, (kernel_size, kernel_size)
        ).sum(axis=(2, 3))
        max_y, max_x = np.unravel_index(np.argmax(density), density.shape)

        # Define the crop window based on max density location
        window = rasterio.windows.Window(max_x, max_y, kernel_size, kernel_size)

        # Crop the image and mask
        cropped_image = src.read(window=window)
        cropped_mask = mask[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ]

        # Keep bands 1 to 10
        return cropped_image[0:10], cropped_mask


def are_all_masks_same(result):
    """
    Checks if all `cropped_mask` arrays in the list are identical.

    Parameters:
    result (list): A list of dictionaries, each containing a 'cropped_mask' key with a NumPy array.

    Returns:
    bool: True if all `cropped_mask` arrays are the same, False otherwise.
    """
    if not result:
        return True  # Empty list, considered as all the same

    # Get the first mask for comparison
    first_mask = result[0]["cropped_mask"]

    # Check if all entries have the same `cropped_mask`
    return all(np.array_equal(entry["cropped_mask"], first_mask) for entry in result)


def main(field_id):
    csv = pd.read_csv("../csvs/fields.csv")
    polygon_wkt = csv[csv["field_id"] == field_id]["polygon"].values[0]
    polygon = wkt.loads(polygon_wkt)
    polygon = Polygon([(y, x) for x, y in polygon.exterior.coords])
    directory = f"../tiffs/{field_id}.0"
    folders = get_folders_with_files(directory)
    result = []
    for folder in folders:
        time = extract_time_from(folder, directory)
        cropped_image, cropped_mask = transform_mask(
            os.path.join(directory, folder, "response.tiff"), polygon
        )
        # Ass the cropped_imag, and the cropped_mask tp the json file
        result.append(
            {
                "folder": folder,
                "time": time,
                "cropped_image": cropped_image,
                "cropped_mask": cropped_mask,
            }
        )

    with open(f"../pickles/{field_id}.pkl", "wb") as pickle_file:
        pickle.dump(result, pickle_file)
        print("Saved", field_id)


# Read the CSV file
# df = pd.read_csv("fields.csv")
# unique_field_ids = df["point_id"].unique()

unique_field_ids = [3979, 9802]

for field_id in unique_field_ids:
    main(field_id)
