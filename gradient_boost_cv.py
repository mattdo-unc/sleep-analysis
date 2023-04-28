import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut


class XGBoostCV:
    def __init__(self, data, n_estimators=256, learning_rate=1e-2, max_depth=3, resample=False, pca=False):
        self.data = data
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
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
            pca = PCA(n_components=0.70)
            X = pca.fit_transform(X)

        # Perform Leave-One-Group-Out cross-validation
        logo = LeaveOneGroupOut()
        metrics = []

        for train_index, test_index in logo.split(X, y, groups):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the XGBoost Classifier
            xgb_clf = XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

            xgb_clf.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = xgb_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_mat = confusion_matrix(y_test, y_pred)

            metrics.append((accuracy, precision, recall, f1, conf_mat))

        mean_metrics = np.mean(metrics, axis=0)
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_conf_mat = mean_metrics

        print("Performance metrics (Leave One Group Out CV):")
        print(f"  - Accuracy: {mean_accuracy:.4f}")
        print(f"  - Precision (weighted): {mean_precision:.4f}")
        print(f"  - Recall (weighted): {mean_recall:.4f}")
        print(f"  - F1-score (weighted): {mean_f1:.4f}")
        print(f"  - Confusion matrix:\n{mean_conf_mat}")
