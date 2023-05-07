from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


class GradientBoost:
    def __init__(self, data, n_estimators=256, learning_rate=1e-2, max_depth=3, pca=False):
        self.data = data
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.pca = pca

    def train_and_display(self):
        X = self.data.iloc[:, 2:]
        y = self.data.iloc[:, 0]

        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Apply PCA for dimensionality reduction
        if self.pca:
            pca = PCA(n_components=0.95)
            X = pca.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Gradient Boosting Classifier
        gb_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )

        gb_clf.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = gb_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if self.pca:
            print(f"Test accuracy (PCA + Gradient Boosting): {accuracy:.4f}")
        else:
            print(f"Test accuracy (Gradient Boosting): {accuracy:.4f}")