import geopandas as gpd
import requests
import json
from shapely.geometry import mapping
import time

# Replace with your Planet API Key

API_KEY = ""


# Function to authenticate with Planet API
def authenticate():
    session = requests.Session()
    session.auth = (API_KEY, "")
    return session


# Function to create an Area of Interest (AOI) from a polygon
def create_aoi(polygon):
    # Convert to GeoJSON format
    print(mapping(polygon)["coordinates"])
    return {"type": "Polygon", "coordinates": mapping(polygon)["coordinates"]}


# Function to search and download satellite images from Planet API
def download_planet_images(aoi):
    # Define the item types and asset types
    item_types = ["PSScene"]
    asset_types = ["ortho_analytic_4b", "ortho_udm2"]

    # Set the date range filter (modify dates as needed)
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": "2017-01-01T00:00:00.000Z",
            "lte": "2017-12-31T23:59:59.999Z",
        },
    }

    # Create the main filter with both geometry and date range filters
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": aoi,
    }

    combined_filter = {"type": "AndFilter", "config": [geometry_filter, date_filter]}

    # Construct the search request
    search_request = {"item_types": item_types, "filter": combined_filter}

    # Make the API request to search for images
    session = authenticate()
    search_url = "https://api.planet.com/data/v1/quick-search"
    headers = {"Content-Type": "application/json"}
    response = session.post(search_url, headers=headers, json=search_request)
    print(response.json())

    if response.status_code != 200:
        print(f"Failed to search images: {response.status_code}, {response.text}")
        return

    search_results = response.json()
    images = search_results.get("features", [])

    if not images:
        print("No images found for the given AOI and date range.")
        return

    # Download images for each result
    for i, image in enumerate(images):
        print(image)
        asset_url = image["_links"]["assets"]
        asset_response = session.get(asset_url)
        if asset_response.status_code != 200:
            print(f"Failed to get assets for image {i+1}: {asset_response.status_code}")
            continue

        assets = asset_response.json()

        # Check and activate the desired assets ("ortho_analytic_4b" and "ortho_udm2")
        for asset_type in asset_types:
            if asset_type in assets:
                activation_url = assets[asset_type]["_links"]["activate"]
                # Activate the asset
                activation_response = session.post(activation_url)
                if activation_response.status_code == 204:
                    print(f"Asset {asset_type} for image {i+1} is now being activated.")
                elif activation_response.status_code == 202:
                    print(f"Asset {asset_type} for image {i+1} is already activated.")
                else:
                    print(
                        f"Failed to activate asset {asset_type} for image {i+1}: {activation_response.status_code}"
                    )
                    continue

                # Wait for asset activation
                asset_ready = False
                while not asset_ready:
                    asset_status_response = session.get(asset_url)
                    if asset_status_response.status_code != 200:
                        print(
                            f"Failed to check asset status for image {i+1}: {asset_status_response.status_code}"
                        )
                        break
                    assets_status = asset_status_response.json()
                    print
                    if assets_status[asset_type]["status"] == "active":
                        asset_ready = True
                        download_url = assets_status[asset_type]["location"]
                        # Download the asset
                        image_response = session.get(download_url)
                        if image_response.status_code == 200:
                            # Save the asset
                            with open(
                                f"planet_image_{i+1}_{asset_type}.tif", "wb"
                            ) as file:
                                file.write(image_response.content)
                            print(f"Downloaded {asset_type} for image {i+1}")
                        else:
                            print(
                                f"Failed to download {asset_type} for image {i+1}: {image_response.status_code}"
                            )
                    else:
                        print(
                            f"Waiting for asset {asset_type} for image {i+1} to become active..."
                        )
                        time.sleep(10)  # Wait before checking again
            else:
                print(f"Asset type {asset_type} is not available for image {i+1}")


# Main function to process the shapefile and download images
def main():
    # Replace 'path/to/your/shapefile.shp' with the path to your shapefile
    shapefile_path = "showcase/shp/fields17.shp"

    # Load the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)

    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Extract the first polygon (you can modify to handle multiple polygons if needed)
    polygon = gdf["geometry"].iloc[0]
    aoi = create_aoi(polygon)

    # Download images for the AOI
    download_planet_images(aoi)


if __name__ == "__main__":
    main()
