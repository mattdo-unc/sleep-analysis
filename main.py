import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="SLEEP STAGE ANALYSIS")
parser.add_argument("--data", type=str, default="static_R_vector_with_ID.csv", help="Path to the data file")
parser.add_argument("--model", type=str, default="gb", help="Model to use for classification")
parser.add_argument("--stages", type=str, default="0,1,2,3", help="Comma-separated list of sleep stages to analyze")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the model (DPL only)")
parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth of the decision tree")
parser.add_argument("--estimators", type=int, default=256, help="Number of estimators for the model")
parser.add_argument("--pca", dest="pca", action="store_true", help="Whether to apply PCA (does not apply in DPL)")
parser.set_defaults(pca=False)
args = parser.parse_args()


def main(params):
    # Load the dataset
    data = pd.read_csv(args.data)

    # Filter data based on the selected sleep stages
    selected_stages = list(map(int, params.stages.split(',')))
    data = data[data.iloc[:, 0].isin(selected_stages)]

    print("Data loaded successfully. Running the model...")

    if params.model == "gb":
        from gradient_boost import GradientBoost
        gb = GradientBoost(data,
                           n_estimators=params.estimators,
                           learning_rate=params.lr,
                           max_depth=params.max_depth,
                           pca=params.pca
                           )
        gb.train_and_display()
    elif params.model == "gbcv":
        from gradient_boost_cv import XGBoostCV
        gb = XGBoostCV(data,
                       learning_rate=params.lr,
                       max_depth=params.max_depth,
                       pca=params.pca
                       )
        gb.train_and_display()
    elif params.model == "svm":
        from svm import SVM
        svm = SVM(data,
                  pca=params.pca
                  )
        svm.train_and_display()
    elif params.model == "svmcv":
        from svm_cv import SVMCV
        svmcv = SVMCV(data, resample=params.resample, pca=params.pca)
        svmcv.train_and_display()
    elif params.model == "rf":
        from random_forest import RFClassifier
        rf = RFClassifier(data,
                          n_estimators=params.estimators,
                          max_depth=params.max_depth,
                          pca=params.pca
                          )
        rf.train_and_display()
    elif params.model == "rfcv":
        from random_forest_cv import RFClassifierCV
        rf = RFClassifierCV(data,
                            n_estimators=params.estimators,
                            max_depth=params.max_depth,
                            pca=params.pca
                            )
        rf.train_and_display()
    else:
        raise ValueError("Invalid model name. Please choose either 'gb' or 'svm' or 'dpl.")


if __name__ == "__main__":
    main(args)
