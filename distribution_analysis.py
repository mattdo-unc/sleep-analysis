import argparse
import pandas as pd
import matplotlib.pyplot as plt


def display_data_distribution(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Get the sleep stage distribution
    stage_counts = data.iloc[:, 0].value_counts()

    # Create a bar plot to display the distribution
    plt.bar(stage_counts.index, stage_counts.values)
    plt.xlabel("Sleep Stages")
    plt.ylabel("Number of Instances")
    plt.title("Sleep Stage Data Distribution")
    plt.xticks(stage_counts.index)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLEEP STAGE DATA DISTRIBUTION")
    parser.add_argument("--data", type=str, default="static_R_vector_with_ID.csv", help="Path to the data file")
    args = parser.parse_args()

    display_data_distribution(args.data)
