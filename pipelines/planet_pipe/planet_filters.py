# the geo json geometry object we got from geojson.io
geo_json_geometry = {
    "type": "Polygon",
    "coordinates": [
        [
            [-98.260496, 19.876792],
            [-98.260781, 19.87756],
            [-98.260351, 19.877841],
            [-98.259754, 19.877259],
            [-98.259849, 19.876936],
            [-98.259627, 19.876369],
            [-98.25929, 19.87564],
            [-98.259301, 19.87537],
            [-98.259811, 19.875959],
            [-98.260195, 19.876415],
            [-98.260496, 19.876792],
        ]
    ],
}
# filter for items the overlap with our chosen geometry
geometry_filter = {
    "type": "GeometryFilter",
    "field_name": "geometry",
    "config": geo_json_geometry,
}

# filter images acquired in a certain date range
date_range_filter = {
    "type": "DateRangeFilter",
    "field_name": "acquired",
    "config": {"gte": "2016-07-01T00:00:00.000Z", "lte": "2016-08-01T00:00:00.000Z"},
}

# filter any images which are more than 50% clouds
cloud_cover_filter = {
    "type": "RangeFilter",
    "field_name": "cloud_cover",
    "config": {"lte": 0.5},
}

# create a filter that combines our geo and date filters
# could also use an "OrFilter"
redding_reservoir = {
    "type": "AndFilter",
    "config": [geometry_filter, date_range_filter, cloud_cover_filter],
}
