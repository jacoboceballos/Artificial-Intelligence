import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

def process_group(file_name, group_name):
    print(f"\nProcessing {group_name}...")
    
    # Load the data
    data = np.loadtxt(file_name, delimiter=',')
    print(f"Loaded {data.shape[0]} samples for {group_name}.")

    # Extract features and labels
    X = data[:, :2]  # First two columns are features (cost and weight)
    y_true = data[:, 2].astype(int)  # Last column is the true label (car type)

    # Use Logistic Regression for classification
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(cm_normalized, index=['Small Car', 'Big Car'], 
                         columns=['Pred Small', 'Pred Big'])

    # Plot the confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Normalized Confusion Matrix - {group_name}')
    plt.ylabel('True Car Type')
    plt.xlabel('Predicted Car Type')
    plt.savefig(f'confusion_matrix_{group_name}.png')
    plt.close()  # Close the plot to free memory
    print(f"Confusion matrix plot saved as 'confusion_matrix_{group_name}.png'")

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Small Car', 'Big Car'])
    with open(f'classification_report_{group_name}.txt', 'w') as f:
        f.write(f"Classification Report for {group_name}\n\n")
        f.write(report)
    print(f"Classification report saved as 'classification_report_{group_name}.txt'")

    # Calculate feature importance
    feature_importance = clf.coef_[0]
    feature_names = ['Cost (USD)', 'Weight (lbs)']

    return cm_df, report, feature_importance, feature_names

# Process each group
groups = [
    ('normalized_groupA.txt', 'Group A'),
    ('normalized_groupB.txt', 'Group B'),
    ('normalized_groupC.txt', 'Group C')
]

results = {}

for file_name, group_name in groups:
    cm_df, report, feature_importance, feature_names = process_group(file_name, group_name)
    results[group_name] = {
        'confusion_matrix': cm_df, 
        'report': report, 
        'feature_importance': feature_importance,
        'feature_names': feature_names
    }

# Generate a combined report
print("\nGenerating combined report...")
with open('combined_report.txt', 'w') as f:
    for group_name, data in results.items():
        f.write(f"\n\n{'='*50}\n")
        f.write(f"Results for {group_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(data['confusion_matrix'].to_string())
        f.write("\n\nClassification Report:\n")
        f.write(data['report'])
        f.write("\nFeature Importance:\n")
        for name, importance in zip(data['feature_names'], data['feature_importance']):
            f.write(f"{name}: {importance:.4f}\n")

print("Combined report saved as 'combined_report.txt'")
print("\nScript execution completed. Please check the output files.")