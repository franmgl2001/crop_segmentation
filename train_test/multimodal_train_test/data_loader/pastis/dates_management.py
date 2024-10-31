import numpy as np
from datetime import datetime, timedelta

def get_features_by_id(data, feature_id):
    """
    Returns the feature with the given ID from the JSON data.
    
    Args:
        data (dict): Parsed JSON data containing feature collection.
        feature_id (str): ID of the feature to search for.
    
    Returns:
        dict: The feature with the matching ID or None if not found.
    """
    for feature in data.get("features", []):
        if feature.get("id") == feature_id:
            return feature
    return None

def number_dates_by_difference(dates_dict, start_date="20180917"):
    """
    Returns a 1D NumPy array with the number of days between each date
    in the input dictionary and the starting date.
    
    Args:
        dates_dict (dict): Dictionary of dates where values are in YYYYMMDD format.
        start_date (str): The reference start date in the format YYYYMMDD.
    
    Returns:
        np.ndarray: 1D array with the day differences.
    """
    # Parse the start date
    start_date_obj = datetime.strptime(start_date, "%Y%m%d")

    # Compute day differences and store in a list
    day_differences = [
        (datetime.strptime(str(value), "%Y%m%d") - start_date_obj).days
        for value in dates_dict.values()
    ]

    # Convert the list to a NumPy array and return
    return np.array(day_differences)