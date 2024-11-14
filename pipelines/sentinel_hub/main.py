from functions.pipe_images import process_sentinel_images
from functions.sentinel_hub_request import download_sentinel_images_in_range
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sentinelhub import SHConfig

csv = pd.read_csv("FASTFARM/fields_cim_1.csv")
csv["year"] = csv["year"].astype(int)

# For loop all the rows in the csv
for index, row in csv.iterrows():

    print(
        f"Processing image for point {row['point_id']} at ({row['point_lat']}, {row['point_long']})",
        f"from {row['year'] - 1}-11-05 to {row['year']}-11-05",
    )
    process_sentinel_images(
        (row["point_lat"], row["point_long"]),
        f"dev_{row['point_id']}",
        area_hectares=10,
        start_date=f"{int(row['year']) -1}-11-05",
        end_date=f"{int(row['year'])}-11-05",
    )


def process_images_parallel(csv_file):
    # Read the CSV file containing the points
    csv = pd.read_csv(csv_file)

    # Use ThreadPoolExecutor to parallelize the processing of images
    with ThreadPoolExecutor(
        max_workers=5
    ) as executor:  # Adjust `max_workers` based on your system's capabilities
        futures = [
            executor.submit(
                process_sentinel_images,
                (row["point_lat"], row["point_long"]),
                row["point_id"],
                area_hectares=40,
                start_date="2022-01-01",
                end_date="2022-12-31",
            )
            for _, row in csv.iterrows()
        ]

        # Wait for all futures to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")


# Example usage
# process_images_parallel("Lucas_italy/Lucas_points_2022.csv")
