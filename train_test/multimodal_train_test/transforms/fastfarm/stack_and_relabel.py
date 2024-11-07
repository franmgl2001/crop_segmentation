import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def extract_field_id_and_year(filename):
    # Remove the file extension and split by underscore
    base_name = filename.split(".")[0]
    parts = base_name.split("_")

    # Extract field_id and year
    field_id = int(parts[0])
    year = int(parts[1])

    return field_id, year


def stack_images_by_time(pickle_file_path):
    # Load the pickle file
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)

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


def get_fielduse_count(field_id):
    fielduses = pd.read_csv("../csvs/fielduses.csv")
    fields_fielduses = fielduses[fielduses["field_id"] == field_id]
    start_date = pd.to_datetime(f"{year-1}-11-05").tz_localize(None)
    end_date = pd.to_datetime(f"{year}-11-05").tz_localize(None)
    fields_fielduses["prediction"] = pd.to_datetime(fields_fielduses["prediction"])

    fields_fielduses = fields_fielduses[
        (fields_fielduses["prediction"] >= start_date)
        & (fields_fielduses["prediction"] <= end_date)
    ]

    # Get the fielduse count
    return len(fields_fielduses)


def relabel_mask(mask, field_id):
    fielduse_count = get_fielduse_count(field_id)
    mask = mask.astype(int)
    mask[mask == 1] = fielduse_count

    return mask


def get_time_keys(data):
    return [datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ") for entry in data]


def create_days(data, year):
    start_date = pd.to_datetime(f"{year-1}-11-05").tz_localize(None)
    time_keys = get_time_keys(data)
    days = [(time_key - start_date).days for time_key in time_keys]
    return days


key = "3979_2023.pickle"


with open(key, "rb") as f:
    data = pickle.load(f)

print(f"Processing file: {key}")
# Extract field_id and year from the key
field_id, year = extract_field_id_and_year(key)
sits_array = stack_images_by_time(key)

# Relabel the mask based on field_id
relabeled_mask = relabel_mask(data[0]["cropped_mask"], field_id)

# Ge the time keys
time_key = [datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ") for entry in data]


days = create_days(data, year)


# Checks plot the relabeled mask, and print the days
plt.imshow(data[0]["cropped_mask"])
# Save the relabeled mask as a numpy file png
plt.savefig("relabeled_mask.png")

print(days)


# Stack the images by time
