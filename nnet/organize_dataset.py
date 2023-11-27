import os
import pandas as pd

# Define the path to your audio files directory and spreadsheet file
audio_directory = 'K:/Thesis/labeled_dataset'
spreadsheet_file = 'K:/Thesis/data_sheet_orig.xlsx'

# Load the spreadsheet into a pandas DataFrame, specifying header row
df = pd.read_excel(spreadsheet_file, header=9)  # Start reading from row 10 (0-based index)

# Create a dictionary to store the mapping of file numbers to labels
file_number_to_label = {}

# Loop through the files in the audio directory
for filename in os.listdir(audio_directory):
    if filename.endswith('.wav'):  # Adjust the file extension as needed
        # Extract the file number and labels from the filename
        parts = filename[:-4].split('. ')  # Remove ".wav" and then split
        file_number = int(parts[0])
        labels = ', '.join(parts[1:])
        
        # Print for debugging
        print(f"File Number: {file_number}, Labels: {labels}")

        # Store the mapping in the dictionary
        file_number_to_label[file_number] = labels

# Create a new column 'Labels' in the DataFrame and populate it with labels
df['Labels'] = df['File No.'].map(file_number_to_label)

# Save the updated DataFrame to a new spreadsheet
df.to_excel('K:/Thesis/data_sheet_new.xlsx', index=False)
