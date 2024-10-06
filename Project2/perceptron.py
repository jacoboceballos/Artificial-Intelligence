import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=5000, activation='hard', gain=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activation = activation
        self.gain = gain
        self.weights = None
        self.bias = None

    def hard_activation(self, x):
        return np.where(x >= 0.5, 1, 0)

    def soft_activation(self, x):
        return 1 / (1 + np.exp(-self.gain * x))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        if self.activation == 'hard':
            return self.hard_activation(linear_output)
        else:
            # Apply a threshold to convert soft activation output to binary
            return np.where(self.soft_activation(linear_output) >= 0.5, 1, 0)

    def fit(self, X, y, epsilon):
        self.weights = np.random.uniform(-0.5, 0.5, X.shape[1])
        self.bias = np.random.uniform(-0.5, 0.5)

        for iteration in range(self.max_iter):
            predictions = self.predict(X)
            errors = y - predictions
            
            # Check total error
            total_error = np.sum(np.abs(errors))
            if total_error < epsilon:
                print(f"Converged after {iteration} iterations.")
                break

            # Update weights and bias
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * np.sum(errors)

            # Show progress every 500 iterations
            if iteration % 500 == 0 or iteration == self.max_iter - 1:
                print(f"Iteration {iteration}: Total Error = {total_error}")

        if iteration == self.max_iter - 1:
            print(f"Reached maximum iterations ({self.max_iter}). Total Error = {total_error}")

        return total_error

def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Cost')
    plt.ylabel('Weight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate rates
    true_positive = cm[1, 1]
    false_positive = cm[0, 1]
    true_negative = cm[0, 0]
    false_negative = cm[1, 0]
    total = np.sum(cm)

    print(f"True Positive Rate: {true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0:.2f}")
    print(f"False Positive Rate: {false_positive / (false_positive + true_negative) if false_positive + true_negative > 0 else 0:.2f}")

# Load datasets
dataset_files = ['normalized_groupA.csv', 'normalized_groupB.csv', 'normalized_groupC.csv']
epsilon_values = {
    'normalized_groupA.csv': 10**-5,
    'normalized_groupB.csv': 40,
    'normalized_groupC.csv': 700,
}

activation_functions = ['hard', 'soft']

for dataset_file in dataset_files:
    print(f"Loading dataset: {dataset_file}")
    data = pd.read_csv(dataset_file)
    X = data[['cost', 'weight']].values
    y = data['type'].values

    for activation in activation_functions:
        print(f"\nTraining on {dataset_file} with {activation} activation function")

        # 75% training and 25% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Print training and testing percentages
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        total_size = train_size + test_size
        print(f"Training Size: {train_size} ({(train_size / total_size) * 100:.2f}%), "
              f"Testing Size: {test_size} ({(test_size / total_size) * 100:.2f}%)")

        # Create and train the model
        model = Perceptron(learning_rate=0.01, activation=activation)
        total_error = model.fit(X_train, y_train, epsilon_values[dataset_file])
        print(f"Final Total Error (75% training): {total_error}\n")

        # Plot decision boundary for training data
        plot_decision_boundary(X_train, y_train, model, f"{dataset_file} Training (75%) with {activation} Activation")

        # Evaluate on the testing set
        evaluate_model(model, X_test, y_test)

        # Repeat with 25% for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)
        
        # Print training and testing percentages
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        total_size = train_size + test_size
        print(f"Training Size: {train_size} ({(train_size / total_size) * 100:.2f}%), "
              f"Testing Size: {test_size} ({(test_size / total_size) * 100:.2f}%)")

        model = Perceptron(learning_rate=0.01, activation=activation)
        total_error = model.fit(X_train, y_train, epsilon_values[dataset_file])
        print(f"Final Total Error (25% training): {total_error}\n")
        
        # Plot decision boundary for training data
        plot_decision_boundary(X_train, y_train, model, f"{dataset_file} Training (25%) with {activation} Activation")

        # Evaluate on the testing set
        evaluate_model(model, X_test, y_test)