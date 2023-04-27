import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from tensorflow.keras.optimizers import schedules, Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


class DeepLearning:
    def __init__(self, data, learning_rate=1e-3, epochs=100, resample=False):
        self.data = data
        self.resample = resample
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.resample = resample

        self.weight_decay = 1e-4

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

        # One-hot encode labels
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = Sequential([
            Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(y_encoded.shape[1], activation='softmax')
        ])

        initial_learning_rate = self.learning_rate
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.09, staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=self.epochs, batch_size=32,
                            callbacks=[early_stopping])

        # Plot the training and validation loss
        self.plot_history(history)

    @staticmethod
    def plot_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(history.history['accuracy'], label='train')
        ax1.plot(history.history['val_accuracy'], label='validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.set_title('Training and Validation Accuracy')

        ax2.plot(history.history['loss'], label='train')
        ax2.plot(history.history['val_loss'], label='validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Training and Validation Loss')

        plt.show()
