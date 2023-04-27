import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut


class SVMCV:
    def __init__(self, data, resample=False, pca=False):
        self.data = data
        self.resample = resample
        self.pca = pca

    def train_and_display(self):
        X = self.data.iloc[:, 2:]
        y = self.data.iloc[:, 0]
        groups = self.data.iloc[:, 1]

        # Apply SMOTE to balance the class distribution
        if self.resample:
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)

        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Apply PCA for dimensionality reduction
        if self.pca:
            pca = PCA(n_components=0.95)
            X = pca.fit_transform(X)

        logo = LeaveOneGroupOut()
        accuracies = []

        for train_index, test_index in logo.split(X, y, groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the SVM
            svm = SVC()
            svm.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        if self.pca:
            print(f"Test accuracy (PCA + SVM + Leave One Group Out CV): {mean_accuracy:.4f}")
        else:
            print(f"Test accuracy (SVM + Leave One Group Out CV): {mean_accuracy:.4f}")
