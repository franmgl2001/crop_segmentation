import requests
import json
import os


# Function to download images from Planet Labs locally
def download_images_locally(
    geometry: dict, start_time: str, end_time: str, product_name: str
):
    """
    Downloads images from Planet Labs locally using the specified geometry, time range, and product name.

    Parameters:
    geometry (dict): The geometry for the area of interest (AOI) in GeoJSON format.
    start_time (str): The start date in 'YYYY-MM-DDTHH:MM:SSZ' format.
    end_time (str): The end date in 'YYYY-MM-DDTHH:MM:SSZ' format.
    product_name (str): Name to use when saving the downloaded files.

    Returns:
    list: A list of downloaded image file paths.
    """

    # Set up your Planet API key
    planet_api_key = ""
    headers = {
        "Authorization": f"api-key {planet_api_key}",
        "Content-Type": "application/json",
    }

    # Define the subscription request
    sub_request = {
        "name": product_name + " ortho+udm",
        "source": {
            "type": "catalog",
            "parameters": {
                "geometry": geometry,
                "start_time": start_time,
                "end_time": end_time,
                "item_types": ["PSScene"],
                "asset_types": ["ortho_analytic_8b_sr", "ortho_udm2"],
            },
        },
        "tools": [{"type": "clip", "parameters": {"aoi": geometry}}],
        "delivery": {"type": "download"},
    }

    # Send the request to create the subscription
    subscription_url = "https://api.planet.com/compute/ops/orders/v2"
    response = requests.post(
        subscription_url, headers=headers, data=json.dumps(sub_request)
    )

    if response.status_code != 202:
        raise Exception(f"Error in subscription creation: {response.text}")

    # Get the subscription ID
    subscription_result = response.json()
    order_id = subscription_result["id"]

    # Check the status of the order and retrieve download links
    order_status_url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
    downloaded_files = []

    # Poll until the order is completed
    while True:
        status_response = requests.get(order_status_url, headers=headers)
        status_result = status_response.json()

        if status_result["state"] == "success":
            # Get the download URLs for assets
            results = status_result["_links"]["results"]
            for item in results:
                download_url = item["location"]
                file_name = item["name"]
                save_path = f"{product_name}/{file_name}"

                # Create directory if it does not exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Download the file locally
                with requests.get(download_url, stream=True) as r:
                    if r.status_code == 200:
                        with open(save_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        downloaded_files.append(save_path)
                    else:
                        raise Exception(
                            f"Error downloading image from {download_url}: {r.status_code}"
                        )

            break
        elif status_result["state"] in ["failed", "partial"]:
            raise Exception(f"Order failed or incomplete: {status_result['state']}")

    return downloaded_files
