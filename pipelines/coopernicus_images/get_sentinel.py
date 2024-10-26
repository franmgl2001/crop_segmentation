from datetime import date, timedelta
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from dotenv import load_dotenv
import os
from functions import get_keycloak

dotenv_path = "./env/.env"

# Load env/.env file
load_dotenv(dotenv_path=dotenv_path)

coopernicus_user = os.getenv("COOPERNICUS_USER")
coopernicus_password = os.getenv("COOPERNICUS_PASSWORD")


# Get a small area of interest
aoi = "POLYGON ((-114.87753440297955 32.36975298719594, -114.87652084719039 32.37043167609002, -114.87742674521925 32.37164319091313, -114.87848124383434 32.37105615241096, -114.87753440297955 32.36975298719594))"
data_collection = "SENTINEL-2"  # Sentinel satellite


today = date.today()
today_string = today.strftime("%Y-%m-%d")
yesterday = today - timedelta(days=6)
yesterday_string = yesterday.strftime("%Y-%m-%d")


json_ = requests.get(
    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and ContentDate/Start gt {yesterday_string}T00:00:00.000Z and ContentDate/Start lt {today_string}T00:00:00.000Z&$count=True&$top=1000"
).json()
p = pd.DataFrame.from_dict(json_["value"])  # Fetch available dataset
if p.shape[0] > 0:  # If we get data back
    p["geometry"] = p["GeoFootprint"].apply(shape)
    # Convert pandas dataframe to Geopandas dataframe by setting up geometry
    productDF = gpd.GeoDataFrame(p).set_geometry("geometry")
    # Remove L1C dataset if not needed
    # productDF = productDF[~productDF["Name"].str.contains("L1C")]
    print(f" total L2A tiles found {len(productDF)}")
    productDF["identifier"] = productDF["Name"].str.split(".").str[0]
    allfeat = len(productDF)

    if allfeat == 0:  # If L2A tiles are not available in current query
        print(f"No tiles found for {today_string}")
    else:  # If L2A tiles are available in current query
        # download all tiles from server
        for index, feat in enumerate(productDF.iterfeatures()):
            try:
                # Create requests session
                session = requests.Session()
                # Get access token based on username and password
                keycloak_token = get_keycloak(coopernicus_user, coopernicus_password)

                session.headers.update({"Authorization": f"Bearer {keycloak_token}"})
                url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feat['properties']['Id']})/$value"
                response = session.get(url, allow_redirects=False)
                while response.status_code in (301, 302, 303, 307):
                    url = response.headers["Location"]
                    response = session.get(url, allow_redirects=False)
                print(feat["properties"]["Id"])
                file = session.get(url, verify=False, allow_redirects=True)

                with open(
                    f"tiles/{feat['properties']['identifier']}.zip",  # location to save zip from copernicus
                    "wb",
                ) as p:
                    print(feat["properties"]["Name"])
                    p.write(file.content)
            except:
                print("problem with server")
else:  # If no tiles found for given date range and AOI
    print("no data found")
