import os
import glob
import pandas as pd

# Specify the directory containing the xlsx files
directory = 'redundancy'

# List all xlsx files in the directory
xlsx_files = glob.glob(os.path.join(directory, '*.xlsx'))

# Convert each xlsx file to a csv file
for xlsx_file in xlsx_files:
    # Read the xlsx file using pandas
    data = pd.read_excel(xlsx_file, engine='openpyxl')

    # Create the output csv file path
    csv_file = os.path.splitext(xlsx_file)[0] + '.csv'

    # Write the data to the csv file
    data.to_csv(csv_file, index=False)

    print(f"Converted '{xlsx_file}' to '{csv_file}'")