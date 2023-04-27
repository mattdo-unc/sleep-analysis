from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RFClassifier:
    def __init__(self, data, n_estimators=256, max_depth=3, resample=False, pca=False):
        self.data = data
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.resample = resample
        self.pca = pca

    def train_and_display(self):
        X = self.data.iloc[:, 2:]
        y = self.data.iloc[:, 0]

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

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        if self.pca:
            print(f"Test accuracy (PCA + Random Forest): {accuracy:.4f}")
        else:
            print(f"Test accuracy (Random Forest): {accuracy:.4f}")
