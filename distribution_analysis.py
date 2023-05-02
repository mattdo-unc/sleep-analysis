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


def plot_data_distribution(data_path):
    data = pd.read_csv(data_path)

    # Get the unique sleep stages
    sleep_stages = data.iloc[:, 0].unique()

    # Set up the plot
    fig, axs = plt.subplots(len(sleep_stages), figsize=(10, 6 * len(sleep_stages)))

    for idx, stage in enumerate(sleep_stages):
        # Extract the data for the current sleep stage
        stage_data = data[data.iloc[:, 0] == stage].iloc[:, 2:]

        # Create a histogram for the current sleep stage
        axs[idx].hist(stage_data.values.flatten(), bins=50)
        axs[idx].set_title(f"Sleep Stage {stage} Data Value Distribution", pad=20)
        axs[idx].set_xlabel("Data Value")
        axs[idx].set_ylabel("Frequency")

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLEEP STAGE DATA DISTRIBUTION")
    parser.add_argument("--data", type=str, default="static_R_vector_with_ID.csv", help="Path to the data file")
    parser.add_argument("--type", type=str, default="total", help="Distribution type")
    args = parser.parse_args()

    if args.type == "total":
        display_data_distribution(args.data)
    elif args.type == "discrete":
        plot_data_distribution(args.data)
