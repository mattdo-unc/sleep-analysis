import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


# TODO: subject IDs the same, old and new


def read_data(file_name):
    # Read the .xlsx file using pandas
    data = pd.read_excel(file_name, engine='openpyxl')
    return data


def preprocess_features(X, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)
    return X_scaled


def reduce_dimensionality(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced


def main():
    # Read the data from the .xlsx file
    file_name = 'syn.xlsx'
    data = read_data(file_name)

    # Extract features and labels from the dataset
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values
    groups = data.iloc[:, 1].values

    # Preprocess the features
    X = preprocess_features(X, method='standard')

    # Reduce dimensionality using PCA
    X = reduce_dimensionality(X, n_components=0.90)

    # Perform leave-one-subject-out cross-validation
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the XGBoost classifier
        clf = XGBClassifier(use_label_encoder=False,
                            eval_metric='mlogloss',
                            n_estimators=200,
                            max_depth=5,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8)
        clf.fit(X_train, y_train)

        # Test the classifier
        y_pred_i = clf.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(y_pred_i)

    # Calculate and print the accuracy and classification report
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    report = classification_report(y_true, y_pred)
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()
