# Sleep Stage Analysis

The dataset is linked independently as it is too heavy to be stored in this repository.

This project aims to analyze sleep stage data using different machine learning models. It takes input from a CSV file containing sleep stage data and applies one of the following models: Gradient Boosting (GB), Support Vector Machine (SVM), or Deep Learning (DPL). Optionally, you can apply PCA and SMOTE for preprocessing the data.

## Usage


```bash
python your_script.py --data <path_to_data> --model <model> --stages <stages> --lr <learning_rate> --resample <resample> --pca <pca> --epochs <epochs>python your_script.py --data <path_to_data> --model <model> --stages <stages> --lr <learning_rate> --resample <resample> --pca <pca> --epochs <epochs>
```


## Arguments

- `--data`: Path to the data file in CSV format. Default is `static_R_vector.csv`.
- `--model`: Model to use for classification. Choose from `gb` (Gradient Boosting), `svm` (Support Vector Machine), or `dpl` (Deep Learning). Default is `gb`.
- `--stages`: Comma-separated list of sleep stages to analyze. Example: `1,2,3,4`. Default is `1,2,3,4`.
- `--lr`: Learning rate for the model (only applicable for DPL). Default is `2e-3`.
- `--resample`: Whether to resample the data using SMOTE. Set to `True` or `False`. Default is `False`.
- `--pca`: Whether to apply PCA (not applicable for DPL). Set to `True` or `False`. Default is `True`.
- `--epochs`: Number of epochs to train the model (only applicable for DPL). Default is `100`.

## Example

```bash
python your_script.py --data "static_R_vector.csv" --model "gb" --stages "1,2,3,4" --lr 2e-3 --resample True --pca True --epochs 100
```

This example will run the script with the following settings:
- Data file: `static_R_vector.csv`
- Model: Gradient Boosting (GB)
- Sleep stages: 1, 2, 3, and 4
- Learning rate: 2e-3 (not applicable for GB)
- Apply SMOTE resampling: True
- Apply PCA: True
- Number of epochs: 100 (not applicable for GB)

