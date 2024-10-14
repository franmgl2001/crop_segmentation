import os
import requests
from requests.auth import HTTPBasicAuth
import time

# our demo filter that filters by geometry, date and cloud cover
from planet_filters import redding_reservoir

# Search API request object
search_endpoint_request = {"item_types": ["PSScene"], "filter": redding_reservoir}
auth = HTTPBasicAuth("", "")

search_result = requests.post(
    "https://api.planet.com/data/v1/quick-search",
    auth=auth,
    json=search_endpoint_request,
)
# Step 2: Process each image result
for item in search_result.json()["features"]:
    item_id = item["id"]
    item_type = "PSScene"
    print(f"Found item: {item_id}")

    # Step 3: Get available assets for the item
    assets_url = (
        f"https://api.planet.com/data/v1/item-types/{item_type}/items/{item_id}/assets"
    )
    assets_result = requests.get(assets_url, auth=auth)

    if assets_result.status_code != 200:
        print(f"Error fetching assets for {item_id}: {assets_result.status_code}")
        continue

    available_assets = assets_result.json()
    print(f"Available assets for {item_id}: {available_assets.keys()}")

    # Select the available asset type (e.g., basic_analytic_4b)
    asset_type = "basic_analytic_4b"  # You can change this to ortho_analytic_4b, etc.

    if asset_type in available_assets:
        selected_asset = available_assets[asset_type]

        # Step 4: Activate the asset if it's inactive
        if selected_asset["status"] == "inactive":
            activate_url = selected_asset["_links"]["activate"]
            print(f"Activating asset for item: {item_id}")
            activation_response = requests.get(activate_url, auth=auth)

            if activation_response.status_code == 202:
                print(
                    f"Activation for {item_id} in progress, waiting for completion..."
                )

            # Wait until the asset is active (polling the status)
            while selected_asset["status"] != "active":
                time.sleep(10)  # wait 10 seconds before checking again
                assets_result = requests.get(assets_url, auth=auth)
                selected_asset = assets_result.json()[asset_type]
                print(f"Asset status: {selected_asset['status']}")
                if selected_asset["status"] == "active":
                    print(f"Asset {asset_type} for item {item_id} is now active!")
                    break

        # Step 5: Download the asset once it's active
        if selected_asset["status"] == "active":
            download_url = selected_asset["location"]
            print(f"Downloading asset for {item_id} from {download_url}")
            download_response = requests.get(download_url)

            if download_response.status_code != 200:
                print(
                    f"Error downloading asset for {item_id}: {download_response.status_code}"
                )
                continue

            # Save the downloaded file
            file_name = f"{item_id}_{asset_type}.tif"
            with open(file_name, "wb") as f:
                f.write(download_response.content)
            print(f"Downloaded: {file_name}")
    else:
        print(f"{asset_type} asset not available for {item_id}")
