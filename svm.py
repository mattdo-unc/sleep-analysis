from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, data, resample=False, pca=False):
        self.data = data
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
            pca = PCA(n_components=0.90)
            X = pca.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the SVM
        svm = SVC()
        svm.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if self.pca:
            print(f"Test accuracy (PCA + SVM): {accuracy:.4f}")
        else:
            print(f"Test accuracy (SVM): {accuracy:.4f}")
