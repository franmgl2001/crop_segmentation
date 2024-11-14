from functions.pipe_images import process_sentinel_images
from functions.sentinel_hub_request import download_sentinel_images_in_range
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

csv = pd.read_csv("FASTFARM/fields_cim_1.csv")
csv["year"] = csv["year"].astype(int)


# Define your function to process each row
def process_row(row):
    print(
        f"Processing image for point {row['point_id']} at ({row['point_lat']}, {row['point_long']})",
        f"from {row['year'] - 1}-11-05 to {row['year']}-11-05",
    )
    process_sentinel_images(
        (row["point_lat"], row["point_long"]),
        f"dev_{row['point_id']}",
        area_hectares=200,
        start_date=f"{int(row['year']) - 1}-11-05",
        end_date=f"{int(row['year'])}-11-05",
    )

# Use ThreadPoolExecutor with max_workers set to 5
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_row, row) for _, row in csv.iterrows()]

    for future in as_completed(futures):
        future.result()


