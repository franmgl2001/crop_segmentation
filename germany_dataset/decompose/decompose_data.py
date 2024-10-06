import tensorflow as tf
import numpy as np

# Path to the .tfrecord.gz file
file_path = "43757.tfrecord.gz"

# Define the fixed width and height of the images
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 24


# Function to parse the examples from the TFRecord to extract labels and x10 images
def _parse_function_images_labels(proto):
    # Define the feature description for labels and x10 images
    feature_description = {
        "labels/data": tf.io.FixedLenFeature([], tf.string),
        "labels/shape": tf.io.FixedLenFeature([3], tf.int64),
        "x10/data": tf.io.FixedLenFeature([], tf.string),
        "x10/shape": tf.io.FixedLenFeature(
            [4], tf.int64
        ),  # Assume [time, width, height, bands]
    }

    return tf.io.parse_single_example(proto, feature_description)


# Function to convert TFRecord to numpy arrays for labels and x10 images
def tfrecord_to_numpy_images_labels(file_path):
    dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP")
    data_list = []

    for raw_record in dataset:
        parsed_record = _parse_function_images_labels(raw_record)

        # Extract the labels
        labels_data = np.frombuffer(
            parsed_record["labels/data"].numpy(), dtype=np.int64
        )
        labels_shape = parsed_record["labels/shape"].numpy()
        labels_data = labels_data.reshape(labels_shape)

        # Extract the x10 images
        x10_data = np.frombuffer(parsed_record["x10/data"].numpy(), dtype=np.float32)
        x10_shape = parsed_record["x10/shape"].numpy()

        # Debugging print statements
        print("x10_shape from TFRecord:", x10_shape)
        print("x10_data size:", x10_data.size)

        # Dynamically adjust the number of bands if necessary
        time_steps = x10_shape[0]
        num_elements_per_image = IMAGE_WIDTH * IMAGE_HEIGHT

        # Calculate the number of bands based on the data size
        calculated_bands = x10_data.size // (time_steps * num_elements_per_image)
        print(f"Calculated number of bands: {calculated_bands}")

        # Update the x10_shape with the calculated number of bands
        x10_shape[3] = calculated_bands

        # Reshape the x10 data
        if x10_data.size == np.prod(x10_shape):
            x10_data = x10_data.reshape(x10_shape)
        else:
            print(
                f"Warning: Data size ({x10_data.size}) does not match expected size ({np.prod(x10_shape)})."
            )
            continue  # Skip this record if shape mismatch is critical

        # Append to the list
        data_list.append({"labels": labels_data, "x10": x10_data})

    return data_list


# Convert the TFRecord file to numpy arrays for labels and x10 images
data_list = tfrecord_to_numpy_images_labels(file_path)

# Example: Print the first label's shape and image's shape
if data_list:
    print("First label's shape:", data_list[0]["labels"].shape)
    print("First image's shape (x10):", data_list[0]["x10"].shape)
