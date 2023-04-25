import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


def read_data(file_name):
    # Read the .xlsx file using pandas
    data = pd.read_excel(file_name, engine='openpyxl')
    return data


def main():
    # Read the data from the .xlsx file
    file_name = 'syn_net.xlsx'
    data = read_data(file_name)

    # Extract features and labels from the dataset
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values
    groups = data.iloc[:, 1].values

    # Initialize the classifier
    clf = RandomForestClassifier()

    # Perform leave-one-subject-out cross-validation
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Select the best k features using SelectKBest
        k = 5
        select_k_best = SelectKBest(score_func=f_classif, k=k)
        X_train = select_k_best.fit_transform(X_train, y_train)
        X_test = select_k_best.transform(X_test)

        # Train the classifier
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
