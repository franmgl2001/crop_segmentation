from planet_pipe import download_images_locally

# Define a 1-hectare AOI polygon as a GeoJSON object
aoi_geojson = {
    "type": "Polygon",
    "coordinates": [
        [
            [-123.1, 38.0],  # Bottom-left corner
            [-123.099, 38.0],  # Bottom-right corner
            [-123.099, 38.0009],  # Top-right corner
            [-123.1, 38.0009],  # Top-left corner
            [-123.1, 38.0],  # Closing the polygon
        ]
    ],
}

downloaded_images = download_images_locally(
    aoi_geojson, "2024-01-01", "2024-06-30", "PSScene"
)
print(downloaded_images)
