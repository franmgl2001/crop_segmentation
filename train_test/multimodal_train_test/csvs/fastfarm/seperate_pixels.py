import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = "pixel_counts.csv"
data = pd.read_csv(file_path)

# Filter out rows where all pixel values are zero
# Filter columns with names from '1' to '2000'
pixel_columns = [col for col in data.columns if col.isdigit() and 1 <= int(col) <= 2000]

# Filter out rows where all pixel values are zero
filtered_data = data[data[pixel_columns].sum(axis=1) > 0]


filtered_data["path"] = (
    filtered_data["field_id"].astype(str)
    + "_"
    + filtered_data["year"].astype(str)
    + ".pkl"
)

# Perform an 80-20 train-test split
train_data, test_data = train_test_split(filtered_data, test_size=0.2)


# Generate txt files of paths that are used in the CustomDataset class
train_paths = train_data["path"].values
test_paths = test_data["path"].values

with open("train_2.txt", "w") as f:
    for path in train_paths:
        f.write(path + "\n")

with open("test_2.txt", "w") as f:
    for path in test_paths:
        f.write(path + "\n")

# Add column called
