import sys
import os
import numpy as np
import scipy.io
import pandas as pd


def convert_mat_to_csv(mat_file):
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_file)

    # Find the first key containing an ndarray in the .mat file
    for key, value in mat_data.items():
        if isinstance(value, (np.ndarray, pd.DataFrame)):
            data = value
            break

    # Convert the data to a pandas DataFrame
    if data.ndim == 2:
        df = pd.DataFrame(data)
    elif data.ndim == 3:
        # Reshape the 3D array into a 2D array before converting to DataFrame
        reshaped_data = data.reshape(data.shape[0], -1)
        df = pd.DataFrame(reshaped_data)
    else:
        raise ValueError("Unsupported data dimension")

    # Save as a .csv file with the same name but with a .csv extension
    csv_file = os.path.splitext(mat_file)[0] + '.csv'
    df.to_csv(csv_file, index=False)
    print(f"Successfully converted {mat_file} to {csv_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mat_to_csv.py <mat_file>")
    else:
        mat_file = sys.argv[1]
        convert_mat_to_csv(mat_file)
