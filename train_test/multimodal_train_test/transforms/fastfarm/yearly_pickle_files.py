import os
import pickle
from datetime import datetime

def process_pickle_files(input_dir):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(".", "yearly_pickle")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the date ranges
    start_nov_2022 = datetime(2022, 11, 1)
    end_nov_2023 = datetime(2023, 11, 1)
    end_nov_2024 = datetime(2024, 11, 1)
    
    # Iterate through each pickle file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(input_dir, filename)
            
            # Extract field_id from the filename (part before '.pkl')
            field_id = filename.split(".pkl")[0]
            
            # Load the pickle file
            with open(file_path, "rb") as file:
                data = pickle.load(file)
            
            # Sort the data by 'time'
            data.sort(key=lambda x: datetime.strptime(x['time'], "%Y-%m-%dT%H:%M:%SZ"))
            
            # Filter data for each date range
            nov_2022_to_nov_2023 = [
                entry for entry in data 
                if start_nov_2022 <= datetime.strptime(entry['time'], "%Y-%m-%dT%H:%M:%SZ") < end_nov_2023
            ]
            
            nov_2023_to_nov_2024 = [
                entry for entry in data 
                if end_nov_2023 <= datetime.strptime(entry['time'], "%Y-%m-%dT%H:%M:%SZ") < end_nov_2024
            ]
            
            # Save filtered data for November 2022 to November 2023
            output_filename_2023 = f"{field_id}_2023.pickle"
            output_path_2023 = os.path.join(output_dir, output_filename_2023)
            with open(output_path_2023, "wb") as output_file:
                pickle.dump(nov_2022_to_nov_2023, output_file)
                print(f"Saved {output_filename_2023} with {len(nov_2022_to_nov_2023)} entries")
            
            # Save filtered data for November 2023 to November 2024
            output_filename_2024 = f"{field_id}_2024.pickle"
            output_path_2024 = os.path.join(output_dir, output_filename_2024)
            with open(output_path_2024, "wb") as output_file:
                pickle.dump(nov_2023_to_nov_2024, output_file)
                print(f"Saved {output_filename_2024} with {len(nov_2023_to_nov_2024)} entries in {output_dir}")

# Specify your input directory path
input_directory = "../pickles"  # Replace with your actual directory path
process_pickle_files(input_directory)
