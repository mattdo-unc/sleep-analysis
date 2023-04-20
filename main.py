import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="SLEEP STAGE ANALYSIS")
parser.add_argument("--data", type=str, default="static_R_vector.csv", help="Path to the data file")
parser.add_argument("--model", type=str, default="gb", help="Model to use for classification")
parser.add_argument("--stages", type=str, default="1,2,3,4", help="Comma-separated list of sleep stages to analyze")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for the model (DPL only)")
parser.add_argument("--resample", type=bool, default=False, help="Whether to resample the data using SMOTE")
parser.add_argument("--pca", type=bool, default=True, help="Whether to apply PCA (does not apply in DPL)")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model  (DPL only)")
args = parser.parse_args()


def main(params):
    # Load the dataset
    data = pd.read_csv(args.data)

    # Filter data based on the selected sleep stages
    selected_stages = list(map(int, params.stages.split(',')))
    data = data[data.iloc[:, 0].isin(selected_stages)]

    print("Data loaded successfully. Running the model...")

    if params.model == "gb":
        from gradient_boost import GradientBooster
        gb = GradientBooster(data, learning_rate=params.lr, max_depth=5, resample=params.resample, pca=params.pca)
        gb.train_and_display()
    elif params.model == "svm":
        from SVM import SVM
        svm = SVM(data, resample=params.resample, pca=params.pca)
        svm.train_and_display()
    elif params.model == "dpl":
        from dpl import DeepLearning
        dl = DeepLearning(data,
                          learning_rate=params.lr,
                          resample=params.resample,
                          epochs=params.epochs
                          )
        dl.train_and_display()
    else:
        raise ValueError("Invalid model name. Please choose either 'gb' or 'svm' or 'dpl.")


if __name__ == "__main__":
    main(args)
