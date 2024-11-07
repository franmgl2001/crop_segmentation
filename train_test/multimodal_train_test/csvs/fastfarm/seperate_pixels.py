import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = 'pixel_counts.csv'
data = pd.read_csv(file_path)

# Filter out rows where all pixel values are zero
# Assuming 'pixels' columns are all columns after the first one
filtered_data = data[(data['1'] != 0) | (data['2'] != 0)]


filtered_data["path"] = filtered_data["field_id"].astype(str) + "_" + filtered_data["year"].astype(str) + ".pkl"

# Perform an 80-20 train-test split
train_data, test_data = train_test_split(filtered_data, test_size=0.2)


# Generate txt files of paths that are used in the CustomDataset class
train_paths = train_data["path"].values
test_paths = test_data["path"].values

with open("train.txt", "w") as f:
    for path in train_paths:
        f.write(path + "\n")

with open("test.txt", "w") as f:
    for path in test_paths:
        f.write(path + "\n")

# Add column called 