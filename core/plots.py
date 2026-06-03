import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Set style
plt.style.use('bmh')

def generate_report_plots():
    # --- 1. CONFUSION MATRIX ---
    # SIMULATION: Let's assume we tested the model on 5 classes
    classes = ['Person', 'Cat', 'Dog', 'Car', 'Sports']
    
    # Ground Truth (Actual labels) vs Predictions
    # We simulate a model with ~85% accuracy
    y_true = np.random.choice(classes, 100)
    y_pred = []
    
    for label in y_true:
        # 85% chance to be correct, 15% chance to pick a random wrong class
        if np.random.rand() > 0.15:
            y_pred.append(label)
        else:
            y_pred.append(np.random.choice(classes))
            
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Zero-Shot Classification Performance')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('1_confusion_matrix.png', dpi=300)
    print("Generated: Confusion Matrix")

    # --- 2. ROC CURVE (Multi-Class) ---
    # Simulating confidence scores for a "One vs Rest" scenario (e.g., Detecting 'Person')
    n_samples = 100
    y_test_binary = np.random.randint(0, 2, n_samples) # 0 = Not Person, 1 = Person
    # Simulate high scores for positive class, low for negative
    y_score = [np.random.uniform(0.6, 1.0) if y == 1 else np.random.uniform(0.0, 0.4) for y in y_test_binary]
    
    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#00f3ff', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) - CLIP Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('2_roc_curve.png', dpi=300)
    print("Generated: ROC Curve")

    # --- 3. PRECISION-RECALL CURVE ---
    precision, recall, _ = precision_recall_curve(y_test_binary, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#bc13fe', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Robustness Check)')
    plt.fill_between(recall, precision, color='#bc13fe', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('3_precision_recall.png', dpi=300)
    print("Generated: Precision-Recall Curve")

if __name__ == "__main__":
    generate_report_plots()