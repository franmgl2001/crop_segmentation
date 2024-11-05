import math
from datetime import datetime, timedelta

### Bouding Box Calculation ###


def calculate_bounding_box_area(point, area_hectares=8):
    """
    Create a bounding box around a point for use with Sentinel Hub with a specified area in hectares.

    Parameters:
    - point (tuple): A tuple containing the latitude and longitude of the point (latitude, longitude).
    - area_hectares (float): Area in hectares for the bounding box. Default is 5 hectares.

    Returns:
    - list: A list containing the coordinates of the bounding box in the format [minLon, minLat, maxLon, maxLat].
    """
    # Constants
    meters_per_degree = 111320  # Approximate meters per degree of latitude

    # Convert area from hectares to square meters
    area_sqm = area_hectares * 10_000  # 5 hectares to square meters (50,000 m^2)

    # Calculate side length in meters for a square box with the given area
    side_length_m = math.sqrt(area_sqm)  # sqrt(50,000 m^2) for a 5-ha area

    # Convert side length to degrees (longitude and latitude)
    latitude, longitude = point
    side_length_deg_lat = (
        side_length_m / meters_per_degree
    )  # Convert meters to degrees latitude
    side_length_deg_lon = side_length_deg_lat / math.cos(
        math.radians(latitude)
    )  # Adjust for latitude

    # Calculate bounding box coordinates
    min_lon = longitude - side_length_deg_lon / 2
    max_lon = longitude + side_length_deg_lon / 2
    min_lat = latitude - side_length_deg_lat / 2
    max_lat = latitude + side_length_deg_lat / 2

    # Return the bbox in the required format
    return [min_lon, min_lat, max_lon, max_lat]


def create_date_ranges(year):
    """
    Create a list of date ranges for each day in the given year.

    Parameters:
    - year (int): The year for which to create date ranges.

    Returns:
    - list: A list of dictionaries with 'from' and 'to' dates for each day in the year.
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = timedelta(days=1)

    date_ranges = []
    while start_date <= end_date:
        date_range = {
            "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
            "to": start_date.strftime("%Y-%m-%dT23:59:59Z"),
        }
        date_ranges.append(date_range)
        start_date += delta

    return date_ranges


def bbox_to_geojson_polygon(bbox):
    """
    Converts a bounding box into a GeoJSON-like polygon format.

    Parameters:
    - bbox (list): A list containing the bounding box coordinates in [minLon, minLat, maxLon, maxLat].

    Returns:
    - dict: A dictionary representing a GeoJSON-like polygon.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Define the coordinates for the polygon
    coordinates = [
        [
            [min_lon, min_lat],  # Bottom-left corner
            [min_lon, max_lat],  # Top-left corner
            [max_lon, max_lat],  # Top-right corner
            [max_lon, min_lat],  # Bottom-right corner
            [min_lon, min_lat],  # Closing point (same as the first)
        ]
    ]

    # Construct the GeoJSON-like polygon
    geojson_polygon = {"type": "Polygon", "coordinates": coordinates}

    return geojson_polygon


def get_point_geojson(point, area_hectares=5):
    bbox = calculate_bounding_box_area(point, area_hectares)
    return bbox_to_geojson_polygon(bbox)
