import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = "pixel_counts.csv"
data = pd.read_csv(file_path)

# Filter columns with names from '1' to '2000' (assumed to be pixel classes)
pixel_columns = [col for col in data.columns if col.isdigit() and 1 <= int(col) <= 2]

# Filter out rows where all pixel values are zero
filtered_data = data[data[pixel_columns].sum(axis=1) > 0]


# Add a 'path' column for file paths
filtered_data["path"] = (
    filtered_data["field_id"].astype(str)
    + "_"
    + filtered_data["year"].astype(str)
    + ".pkl"
)

# Perform an 80-20 train-test split
train_data, test_data = train_test_split(filtered_data, test_size=0.2)

# Calculate the total number of pixels per class in train and test datasets
train_pixel_counts_per_class = train_data[pixel_columns].sum()
test_pixel_counts_per_class = test_data[pixel_columns].sum()

# Print the pixel counts per class
print("Pixel counts per class in training set:")
print(train_pixel_counts_per_class)

print("\nPixel counts per class in test set:")
print(test_pixel_counts_per_class)

# Save the results as text files (optional)
train_pixel_counts_per_class.to_csv("train_pixel_counts_per_class.csv", header=True)
test_pixel_counts_per_class.to_csv("test_pixel_counts_per_class.csv", header=True)

# Generate txt files of paths that are used in the CustomDataset class
train_paths = train_data["path"].values
test_paths = test_data["path"].values

with open("train_4.txt", "w") as f:
    for path in train_paths:
        f.write(path + "\n")

with open("test_4.txt", "w") as f:
    for path in test_paths:
        f.write(path + "\n")
