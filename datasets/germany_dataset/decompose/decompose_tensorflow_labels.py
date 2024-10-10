import tensorflow as tf
import numpy as np

# Path to the .tfrecord.gz file
file_path = "43757.tfrecord.gz"


# Function to parse the examples from the TFRecord to extract only labels
def _parse_function_labels(proto):
    # Define the feature description for the labels
    feature_description = {
        "labels/data": tf.io.FixedLenFeature([], tf.string),
        "labels/shape": tf.io.FixedLenFeature([3], tf.int64),
    }

    return tf.io.parse_single_example(proto, feature_description)


# Function to convert TFRecord to numpy arrays for labels only
def tfrecord_to_numpy_labels(file_path):
    dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP")
    numpy_labels = []

    for raw_record in dataset:
        parsed_record = _parse_function_labels(raw_record)

        # Extract the labels
        labels_data = np.frombuffer(
            parsed_record["labels/data"].numpy(), dtype=np.int64
        )
        labels_shape = parsed_record["labels/shape"].numpy()

        # Reshape the labels data based on the shape information
        labels_data = labels_data.reshape(labels_shape)

        # Append to the list
        numpy_labels.append(labels_data)

    return numpy_labels


# Convert the TFRecord file to numpy arrays for labels
labels_data = tfrecord_to_numpy_labels(file_path)

# Example: Print the first label's shape
print("First label's shape:", labels_data[0].shape)
print("First label's data:", labels_data[0])
