import pandas as pd

# Define the normalization function
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

# Function to normalize and save data to a csv file
def normalize_and_save_to_csv(input_file, output_file):
    try:
        # Load dataset without headers (comma-separated by default)
        dataset = pd.read_csv(input_file, header=None)
        
        # Assign column names manually
        dataset.columns = ['cost', 'weight', 'type']

        # Normalize the 'cost', 'weight', and 'type' columns
        dataset['cost'] = normalize_column(dataset['cost'])
        dataset['weight'] = normalize_column(dataset['weight'])
        dataset['type'] = normalize_column(dataset['type'])

        # Save the entire dataset as a CSV file (comma-separated)
        dataset.to_csv(output_file, sep=',', index=False, header=True)

        print(f"Normalized data saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except pd.errors.ParserError:
        print(f"Error: Could not parse the file {input_file}. Please check the format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Array of input files and corresponding output csv files (use your absolute paths)
file_names = [
    '/Users/jacobceballos/Desktop/Artificial Intelligence/Project1_Data_F24-2/groupA.txt',
    '/Users/jacobceballos/Desktop/Artificial Intelligence/Project1_Data_F24-2/groupB.txt',
    '/Users/jacobceballos/Desktop/Artificial Intelligence/Project1_Data_F24-2/groupC.txt'
]
output_files = [
    '/Users/jacobceballos/Desktop/Artificial Intelligence/Project1_Data_F24-2/normalized_groupA.csv',
    '/Users/jacobceballos/Desktop/Artificial Intelligence/Project1_Data_F24-2/normalized_groupB.csv',
    '/Users/jacobceballos/Desktop/Artificial Intelligence/Project1_Data_F24-2/normalized_groupC.csv'
]

# Loop through each file, normalize, and save
for i in range(len(file_names)):
    normalize_and_save_to_csv(file_names[i], output_files[i])
