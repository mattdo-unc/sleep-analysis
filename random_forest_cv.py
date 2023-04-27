from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RFClassifierCV:
    def __init__(self, data, n_estimators=256, max_depth=3, resample=False, pca=False):
        self.data = data
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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
            pca = PCA(n_components=0.90)
            X = pca.fit_transform(X)

        # Perform Leave-One-Group-Out cross-validation
        logo = LeaveOneGroupOut()
        accuracies = []

        for train_index, test_index in logo.split(X, y, groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the Random Forest Classifier
            rf_clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )

            rf_clf.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = rf_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)

        if self.pca:
            print(f"\nAverage test accuracy (PCA + Random Forest): {avg_accuracy:.4f}")
        else:
            print(f"\nAverage test accuracy (Random Forest): {avg_accuracy:.4f}")
