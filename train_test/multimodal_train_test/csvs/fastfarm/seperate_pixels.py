import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = 'pixel_counts.csv'
data = pd.read_csv(file_path)

# Filter out rows where all pixel values are zero
# Assuming 'pixels' columns are all columns after the first one
print(data.dtypes)
filtered_data = data[(data['1'] != 0) | (data['2'] != 0)]

# Perform an 80-20 train-test split
train_data, test_data = train_test_split(filtered_data, test_size=0.2)

# Save the train and test sets to separate files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)