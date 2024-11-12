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
from datetime import datetime
import matplotlib.pyplot as plt


relabel_json = {1: 1, 2: 2, 14: 3, 17: 4}


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


def separate_data_by_years(data):
    """
    Separates the data into two lists based on yearly ranges.

    Parameters:
    - data (list): List of dictionaries, each containing a 'time' key in ISO format.

    Returns:
    - tuple: Two lists of dictionaries, one for November 2022 to November 2023 and
             one for November 2023 to November 2024.
    """
    # Define the date ranges
    start_nov_2022 = datetime(2022, 11, 5)
    end_nov_2023 = datetime(2023, 11, 5)
    end_nov_2024 = datetime(2024, 11, 5)

    # Sort the data by 'time'
    data.sort(key=lambda x: datetime.strptime(x["time"], "%Y-%m-%dT%H:%M:%SZ"))

    # Filter data for each date range
    nov_2022_to_nov_2023 = [
        entry
        for entry in data
        if start_nov_2022
        <= datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ")
        < end_nov_2023
    ]
    nov_2023_to_nov_2024 = [
        entry
        for entry in data
        if end_nov_2023
        <= datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ")
        < end_nov_2024
    ]

    return nov_2022_to_nov_2023, nov_2023_to_nov_2024


def stack_images_by_time(data):
    # Load the pickle file

    # Sort the data by the 'time' field to ensure chronological order
    data.sort(key=lambda x: datetime.strptime(x["time"], "%Y-%m-%dT%H:%M:%SZ"))

    # Verify the timestamps are in order
    timestamps = [entry["time"] for entry in data]
    for i in range(1, len(timestamps)):
        if datetime.strptime(timestamps[i], "%Y-%m-%dT%H:%M:%SZ") < datetime.strptime(
            timestamps[i - 1], "%Y-%m-%dT%H:%M:%SZ"
        ):
            print("Error: Timestamps are out of order!")
            return None

    print("Timestamps are in chronological order.")

    # Stack all 'cropped_image' arrays into a single SITS array
    sits_array = np.stack([entry["cropped_image"] for entry in data])

    print(f"Created SITS with shape: {sits_array.shape}")
    return sits_array


def get_fielduse_count(field_id, year):
    return len(get_field_id_fielduses(field_id, year))


def get_field_id_fielduses(field_id, year):
    fielduses = pd.read_csv("../csvs/fielduses.csv")
    fields_fielduses = fielduses[fielduses["field_id"] == field_id]
    start_date = pd.to_datetime(f"{year-1}-11-05").tz_localize(None)
    end_date = pd.to_datetime(f"{year}-11-05").tz_localize(None)
    fields_fielduses["prediction"] = pd.to_datetime(fields_fielduses["prediction"])

    fields_fielduses = fields_fielduses[
        (fields_fielduses["prediction"] >= start_date)
        & (fields_fielduses["prediction"] <= end_date)
    ]

    return fields_fielduses


def relabel_mask(mask, field_id, year):
    fielduse_count = get_fielduse_count(field_id, year)
    mask = mask.astype(int)
    mask[mask == 1] = fielduse_count

    return mask


def relabel_crop_mask(mask, field_id, year):
    fielduses = get_field_id_fielduses(field_id, year)
    mask = mask.astype(int)
    # Relabel to the first fielduse that is not a 4
    for index, row in fielduses.iterrows():
        if row["fielduse_id"] != 4:
            print("Relabeling to ", row["fielduse_id"])
            mask[mask == 1] = relabel_json[str(row["fielduse_id"])]
    return mask


def get_time_keys(data):
    return [datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ") for entry in data]


def create_days(data, year):
    start_date = pd.to_datetime(f"{year-1}-11-05").tz_localize(None)
    time_keys = get_time_keys(data)
    days = [(time_key - start_date).days for time_key in time_keys]
    return days


import numpy as np
import csv
from collections import Counter


def register_pixel_counts(
    mask, field_id, year, num_classes, csv_filename="pixel_counts.csv"
):
    """
    Registers the counts of pixel values in a 2D NumPy array (mask) for a specified number of classes,
    along with field_id and year.

    Parameters:
    - mask (np.ndarray): A 2D NumPy array containing pixel values.
    - field_id (str): Identifier for the field.
    - year (str): Year or date range as a string, e.g., "2023" or "2024".
    - num_classes (int): Number of classes in the mask (e.g., 3 for classes 0, 1, 2).
    - csv_filename (str): The name of the CSV file to write the data to. Defaults to "pixel_counts.csv".
    """
    # Ensure the mask is a 2D array
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D NumPy array")

    # Count occurrences of each class in the mask
    counts = Counter(mask.flatten())
    class_counts = [counts.get(i, 0) for i in range(num_classes)]

    # Write the counts to a CSV file
    with open(csv_filename, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write header if file is empty
        csv_file.seek(0, 2)  # Move to the end of the file
        if csv_file.tell() == 0:
            header = ["field_id", "year"] + [str(i) for i in range(num_classes)]
            writer.writerow(header)

        # Write the data row
        writer.writerow([field_id, year] + class_counts)

    print(f"Registered counts for field_id: {field_id}, year: {year}")


def main(field_id, relabel="Binary", years=[2023, 2024]):
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

    # Seperate the data per year
    results = separate_data_by_years(result)

    for num, res in enumerate(results):
        if len(res) == 0:
            return
        sits_array = stack_images_by_time(res)
        if relabel == "Binary":
            relabeled_mask = relabel_mask(res[0]["cropped_mask"], field_id, years[num])
        else:
            relabeled_mask = relabel_crop_mask(
                res[0]["cropped_mask"], field_id, years[num]
            )

        # Save mask as relabeled png
        days = create_days(res, years[num])

        plt.imshow(relabeled_mask)
        plt.savefig("label.png")
        # Save the relabel mask as png

        register_pixel_counts(relabeled_mask, field_id, years[num])

        # Label the amount of 0, 1, 2 labels in the label

        with open(f"pickles_crops/{field_id}_{years[num]}.pkl", "wb") as f:
            pickle.dump(
                {
                    "image": sits_array,
                    "mask": relabeled_mask,
                    "doy": days,
                },
                f,
            )


# Read the CSV file
df = pd.read_csv("../csvs/fields.csv")
unique_field_ids = df["field_id"].unique()

for field_id in unique_field_ids:
    main(field_id, relabel="crop")
    print(f"Finished processing field_id: {field_id}")
