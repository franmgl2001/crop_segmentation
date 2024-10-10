"""
Get the keys and their data types in a TFRecord file.

By: Francisco Martinez Gallardo
Date: 2024-10-05
"""

import tensorflow as tf

# Path to the .tfrecord.gz file
file_path = "43757.tfrecord.gz"

# Create a dataset from the .tfrecord.gz file
raw_dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP")


# Function to print the keys and their data types in the TFRecord
def inspect_tfrecord(file_path):
    for raw_record in tf.data.TFRecordDataset(file_path, compression_type="GZIP").take(
        1
    ):
        # Parse the record using tf.train.Example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Extract and print the feature keys and their types
        features = example.features.feature
        print("Feature keys and data types in the TFRecord:")
        for key, feature in features.items():
            kind = feature.WhichOneof("kind")
            print(f"Key: {key}, Type: {kind}")


def inspect2_tfrecord(file_path):
    for i, raw_record in enumerate(
        tf.data.TFRecordDataset(file_path, compression_type="GZIP").take(5)
    ):  # Inspect first 5 records
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature.keys()
            print(f"Record {i} feature keys: {features}")
        except Exception as e:
            print(f"Error parsing record {i}: {e}")


# Inspect the TFRecord file


inspect_tfrecord(file_path)
inspect2_tfrecord(file_path)
