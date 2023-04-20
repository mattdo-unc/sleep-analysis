import os
import sys
import scipy.io
import pandas as pd


def mat_to_csv(mat_file):
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file)
    variables = [var_name for var_name in mat_data if not var_name.startswith("__")]
    print(variables)

    # Check if file with name already exists
    for var_name in variables:
        if f'{var_name}.csv' in os.listdir():
            continue
        elif not var_name.startswith("__"):
            data = mat_data[var_name]
            df = pd.DataFrame(data)
            df.to_csv(f'{var_name}.csv', index=False)
            break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mat_to_csv.py <input_mat_file>")
        sys.exit(1)

    input_mat_file = sys.argv[1]

    mat_to_csv(input_mat_file)