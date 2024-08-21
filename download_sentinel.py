import geopandas as gpd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date


def fetch_sentinel_data(geodf, start_date, end_date):
    # Your Sentinel API credentials
    username = "your_username"
    password = "your_password"
    api_url = "https://scihub.copernicus.eu/dhus"

    # Initialize the Sentinel API
    api = SentinelAPI(username, password, api_url)

    # Loop through each geometry in the GeoDataFrame
    for index, row in geodf.iterrows():
        # Convert geometry to WKT
        area_of_interest = row["geometry"].to_wkt()

        # Sentinel-2: Search and download with cloud cover filter
        s2_products = api.query(
            area_of_interest,
            date=(start_date, end_date),
            platformname="Sentinel-2",
            cloudcoverpercentage=(0, 30),
        )

        if s2_products:
            print(
                f"Downloading {len(s2_products)} Sentinel-2 products for AOI {index + 1}"
            )
            api.download_all(s2_products)
        else:
            print(f"No Sentinel-2 products found for AOI {index + 1}")

        # Sentinel-1: Search and download
        s1_products = api.query(
            area_of_interest,
            date=(start_date, end_date),
            platformname="Sentinel-1",
            producttype="GRD",
        )

        if s1_products:
            print(
                f"Downloading {len(s1_products)} Sentinel-1 products for AOI {index + 1}"
            )
            api.download_all(s1_products)
        else:
            print(f"No Sentinel-1 products found for AOI {index + 1}")

    print("Download complete.")


# Example usage
if __name__ == "__main__":
    # Load your polygons as a GeoDataFrame (replace with your file path)
    geodf = gpd.read_file("path_to_your_polygons.shp")

    # Define the time period of interest (all of 2022)
    start_date = "20220101"
    end_date = "20221231"

    # Fetch Sentinel-1 and Sentinel-2 data
    fetch_sentinel_data(geodf, start_date, end_date)
