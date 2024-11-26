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


relabel_json = {
    14: 1,
    17: 2,
    1: 3,
    2: 4,
    18: 5,
    15: 6,
}


class_names = {
    0: "Background",
    1: "Barley",
    2: "Wheat",
    3: "Sorhgum",
    4: "Maize",
}
num_classes = 5


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


def separate_data_by_years(data, dates):
    """
    Separates the data into a dictionary where each key is a year, representing
    data from November 5 of the previous year to November 5 of the current year.

    Parameters:
    - data (list): List of dictionaries, each containing a 'time' key in ISO format.
    - dates (dict): Dictionary containing the start and end dates for the data range.

    Returns:
    - dict: A dictionary where each key is a year (e.g., 2019) and the value is a list of
            dictionaries containing the data for November 5 of the previous year to
            November 5 of the current year.
    """
    # Sort the data by 'time'
    data.sort(key=lambda x: datetime.strptime(x["time"], "%Y-%m-%dT%H:%M:%SZ"))

    # Extract the earliest and latest years from the data
    times = [datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ") for entry in data]
    earliest_year = times[0].year
    latest_year = times[-1].year

    # Create a dictionary to store the separated data
    yearly_data = {}

    # Loop through each year to define the date ranges dynamically
    for year in range(earliest_year, latest_year + 1):
        start_date = datetime(year - 1, dates["start_month"], dates["start_day"])
        end_date = datetime(year, dates["end_month"], dates["end_day"])

        # Filter data for the current year range
        yearly_data[year] = [
            entry
            for entry in data
            if start_date
            <= datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ")
            < end_date
        ]

    # Remove empty entries
    yearly_data = {year: data for year, data in yearly_data.items() if data}
    return yearly_data


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
    print(field_id, year, type(field_id), type(year))
    fielduses = pd.read_csv("../csvs/full_fielduses.csv")
    fielduses["field_id"] = fielduses["field_id"].astype(str)
    fielduses["field_id"] = fielduses["field_id"].str.replace(".0", "")
    fielduses["year"] = fielduses["year"].astype(int, errors="ignore")
    fields_fielduses = fielduses[
        (fielduses["field_id"] == field_id) & (fielduses["year"] == year)
    ]
    return fields_fielduses


def relabel_mask(mask, field_id, year):
    print("Relabeling mask")
    fielduse_count = get_fielduse_count(field_id, year)
    mask = mask.astype(int)
    mask[mask == 1] = fielduse_count

    return mask


def relabel_crop_mask(mask, field_id, year):
    print("Relabeling crop mask", field_id, year)
    fielduses = get_field_id_fielduses(field_id, year)
    mask = mask.astype(int)
    # Relabel to the first fielduse that is not a 4
    for index, row in fielduses.iterrows():
        if row["crop_id"] != 4:
            print(
                "Relabeling to ",
                row["crop_id"],
                "to ",
                relabel_json[row["crop_id"]],
            )
            mask[mask == 1] = relabel_json[row["crop_id"]]
            return mask

    # If no relabeling relabel to 0
    if 1 in mask:
        print("Relabeling to 0")
        mask[mask == 1] = 0
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
        print(len(class_counts))
        writer.writerow([field_id, year] + class_counts)

    print(f"Registered counts for field_id: {field_id}, year: {year}")


def main(field_id, dates, relabel="Binary"):
    csv = pd.read_csv("../csvs/full_fields.csv")
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
    results = separate_data_by_years(result, dates)

    # print(len(results.items()), type(results), print(results.keys()))

    for year in results.keys():
        if len(results[year]) == 0:
            return
        sits_array = stack_images_by_time(results[year])

        if relabel == "Binary":
            relabeled_mask = relabel_mask(
                results[year][0]["cropped_mask"], field_id, year
            )
        else:
            relabeled_mask = relabel_crop_mask(
                results[year][0]["cropped_mask"], field_id, year
            )

        # Save mask as relabeled png
        days = create_days(results[year], year)

        plt.imshow(relabeled_mask)
        plt.savefig("label.png")
        # Save the relabel mask as png

        register_pixel_counts(
            relabeled_mask,
            field_id,
            year,
            num_classes,
        )

        # Label the amount of 0, 1, 2 labels in the label
        print(sits_array.shape, relabeled_mask.shape, len(days))

        with open(f"pickle_crops_yearly/{field_id}_{year}.pkl", "wb") as f:
            pickle.dump(
                {
                    "image": sits_array,
                    "mask": relabeled_mask,
                    "doy": days,
                },
                f,
            )


import concurrent.futures
import pandas as pd

# Read the CSV file
df = pd.read_csv("../csvs/full_fields.csv")
unique_field_ids = df["field_id"].unique()

dates = {"start_day": 1, "start_month": 1, "end_day": 31, "end_month": 12}


def process_field(field_id):
    """
    Wrapper function for processing a single field_id.
    """
    try:
        main(field_id, dates, relabel="crop")
        print(f"Finished processing field_id: {field_id}")
    except Exception as e:
        print(f"Error processing field_id {field_id}: {e}")


# Run in parallel

if __name__ == "__main__":
    # Adjust the number of workers based on your CPU cores
    max_workers = 1  # Example: Use 4 parallel processes

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all field_id processing tasks
        future_to_field = {
            executor.submit(process_field, field_id, dates): field_id
            for field_id in unique_field_ids
        }

        # Monitor progress
        for future in concurrent.futures.as_completed(future_to_field):
            field_id = future_to_field[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Field_id {field_id} generated an exception: {exc}")
