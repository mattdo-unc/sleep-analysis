import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeavePGroupsOut


class RFClassifierCV:
    def __init__(self, data, n_estimators=256, max_depth=3, pca=False):
        self.data = data
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.pca = pca

    def train_and_display(self):
        X = self.data.iloc[:, 2:]
        y = self.data.iloc[:, 0]
        groups = self.data.iloc[:, 1]

        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Apply PCA for dimensionality reduction
        if self.pca:
            pca = PCA(n_components=0.95)
            X = pca.fit_transform(X)
            print(f"Original number of dimensions: {self.data.shape[1] - 2}")
            print(f"Reduced number of dimensions after PCA: {pca.n_components_}")

        # Perform Leave-P-Groups-Out cross-validation
        lpgo = LeavePGroupsOut(n_groups=3)
        metrics = []

        for train_index, test_index in lpgo.split(X, y, groups):
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
            y_proba = rf_clf.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_mat = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')

            metrics.append((accuracy, precision, recall, f1, roc_auc, conf_mat))

        mean_metrics = np.mean(metrics, axis=0)
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, mean_conf_mat = mean_metrics

        print("Performance metrics (Leave P Groups Out CV):")
        print(f"  - Accuracy: {mean_accuracy:.4f}")
        print(f"  - Precision (weighted): {mean_precision:.4f}")
        print(f"  - Recall (weighted): {mean_recall:.4f}")
        print(f"  - F1-score (weighted): {mean_f1:.4f}")
        print(f"  - ROC AUC (OvO): {mean_auc:.4f}")
        print(f"  - Confusion matrix:\n{mean_conf_mat}")
