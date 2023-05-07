# Sleep Stage Analysis

The dataset is [linked independently](https://drive.google.com/file/d/16kcB-uY_y8frkL5KBdtf9u0DZntetqRn/view?usp=sharing) as it is too heavy to be stored in this repository.

This project aims to analyze sleep stage data using different machine learning models. It takes input from a CSV file containing sleep stage data and applies one of the following models: Gradient Boosting (GB), Gradient Boosting with Cross Validation (GBCV), Support Vector Machine (SVM), Support Vector Machine with Cross Validation (SVMCV), Random Forest (RF), or Random Forest with Cross Validation (RFCV). Optionally, you can apply PCA for preprocessing the data. Data is also normalized by default.

Despite having code available for T/T models, our report focuses on the leave-one-group-out cross validation models. The entirety of the project has been kept in this repository for analysis purposes.

## Usage

```bash
python your_script.py --data <path_to_data> --model <model> --stages <stages> --lr <learning_rate> --max_depth <max_depth> --estimators <estimators> --pca <pca> --epochs <epochs>
```

## Arguments

- `--data`: Path to the data file in CSV format. Default is `static_R_vector_with_ID.csv`.
- `--model`: Model to use for classification. Choose from `gb` (Gradient Boosting), `gbcv` (Gradient Boosting with Cross Validation), `svm` (Support Vector Machine), `svmcv` (Support Vector Machine with Cross Validation), `rf` (Random Forest), or `rfcv` (Random Forest with Cross Validation). Default is `gb`.
- `--stages`: Comma-separated list of sleep stages to analyze. Example: `0,1,2,3`. Default is `0,1,2,3`.
- `--lr`: Learning rate for the model (only applicable for GB and GBCV). Default is `1e-2`.
- `--max_depth`: Maximum depth of the decision tree. Default is `3`.
- `--estimators`: Number of estimators for the model. Default is `256`.
- `--pca`: Whether to apply PCA (not applicable for DPL). Set to `True` or `False`. Default is `False`.
- `--epochs`: Number of epochs to train the model (only applicable for DPL). Default is `100`.

View the `--help` command for additional detail.


## Sleep Stage Data Distribution

We wrote the following tool to help visualize the distribution of sleep stages in the given dataset. It reads the sleep stage data from a CSV file and creates a bar plot displaying the number of instances per sleep stage.

### Usage

Run the script `data_distribution.py` and enter the path to the CSV file when prompted:

```bash
python data_distribution.py --data <path_to_data>
```

This will display a bar plot showing the number of instances for each sleep stage in the selected dataset.