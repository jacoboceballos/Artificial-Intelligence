import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import seaborn as sns

def hard_activation(x):
    return 1 if x >= 0 else 0

def soft_activation(x, gain=1):
    return 1 / (1 + np.exp(-gain * x))

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01, max_iterations=5000):
        self.weights = np.random.uniform(-0.5, 0.5, input_size + 1)  # +1 for bias
        self.activation = activation_function
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def predict(self, X):
        return self.activation(np.dot(X, self.weights[1:]) + self.weights[0])

    def train(self, X, y, epsilon):
        iterations = 0
        total_error = float('inf')
        
        while total_error > epsilon and iterations < self.max_iterations:
            total_error = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                total_error += error ** 2
                
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error
            
            iterations += 1
        
        return iterations, total_error

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    X = data[:, :2]  # First two columns (cost and weight)
    y = data[:, 2]   # Third column (car type)
    return X, y

def plot_data_and_decision_boundary(X, y, perceptron, title, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Small Car', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Big Car', alpha=0.7)
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.xlabel('Normalized Cost')
    plt.ylabel('Normalized Weight')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(X_test, y_test, perceptron):
    y_pred = np.array([perceptron.predict(x) for x in X_test])
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)
    
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    return cm, accuracy, tpr, tnr, fpr, fnr

def run_experiment(dataset, activation_func, epsilon, train_size, results_dir):
    X, y = load_data(f'normalized_group{dataset}.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    
    activation_name = "Hard" if activation_func == hard_activation else "Soft"
    exp_name = f"Dataset_{dataset}_{activation_name}_Activation_{train_size*100:.0f}%_Training"
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    perceptron = Perceptron(input_size=2, activation_function=activation_func)
    iterations, final_error = perceptron.train(X_train, y_train, epsilon)
    
    plot_data_and_decision_boundary(X_train, y_train, perceptron, f'Dataset {dataset} - Training Data', 
                                    os.path.join(exp_dir, 'training_plot.png'))
    plot_data_and_decision_boundary(X_test, y_test, perceptron, f'Dataset {dataset} - Testing Data', 
                                    os.path.join(exp_dir, 'testing_plot.png'))
    
    cm, accuracy, tpr, tnr, fpr, fnr = evaluate_model(X_test, y_test, perceptron)
    plot_confusion_matrix(cm, f'Confusion Matrix - Dataset {dataset}', 
                          os.path.join(exp_dir, 'confusion_matrix.png'))
    
    results = {
        "Dataset": dataset,
        "Activation": activation_name,
        "Training Size": f"{train_size*100:.0f}%",
        "Iterations": iterations,
        "Final Total Error": final_error,
        "Accuracy": accuracy,
        "True Positive Rate": tpr,
        "True Negative Rate": tnr,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    }
    
    with open(os.path.join(exp_dir, 'results.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return results

# Main execution
if __name__ == "__main__":
    print("Script started")
    datasets = ['A', 'B', 'C']
    epsilons = {'A': 1e-5, 'B': 40, 'C': 700}
    results_dir = 'experiment_results'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    for dataset in datasets:
        print(f"Processing Dataset {dataset}")
        try:
            for activation_func in [hard_activation, soft_activation]:
                for train_size in [0.75, 0.25]:
                    results = run_experiment(dataset, activation_func, epsilons[dataset], train_size, results_dir)
                    all_results.append(results)
                    print(f"Completed: Dataset {dataset}, {'Hard' if activation_func == hard_activation else 'Soft'} Activation, {train_size*100:.0f}% Training")
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
    
    # Save overall results
    with open(os.path.join(results_dir, 'overall_results.txt'), 'w') as f:
        for result in all_results:
            f.write(str(result) + '\n\n')

    print("Script completed")