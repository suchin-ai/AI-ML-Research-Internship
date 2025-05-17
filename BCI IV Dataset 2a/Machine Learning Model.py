import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Load the Data ===
X = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_X_CSD.npy')
y = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_y_CSD.npy')

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

# === 2. Flatten the EEG epochs ===
X_flat = X.reshape(X.shape[0], -1)  # (2592, 22*501) => (2592, 11022)
print(f"Flattened X shape: {X_flat.shape}")

# === 3. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# === 4. Initialize Models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, solver='saga', multi_class='multinomial', n_jobs=-1),
    "Support Vector Machine": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# === 5. Train and Evaluate ===
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{model_name} Accuracy: {acc*100:.2f}%")
    print(f"{model_name} F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    results.append({"Model": model_name, "Accuracy (%)": acc*100, "F1-Score": f1})

# === 6. Summarize Results ===
results_df = pd.DataFrame(results)
print("\n=== Model Comparison Table ===")
print(results_df)
# === Plot Accuracy and F1-Score Comparison ===
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax[0].bar(results_df["Model"], results_df["Accuracy (%)"], color='skyblue')
ax[0].set_title('Model Accuracy Comparison')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_ylim(0, 100)
ax[0].set_xticklabels(results_df["Model"], rotation=45, ha="right")

# F1-Score plot
ax[1].bar(results_df["Model"], results_df["F1-Score"], color='salmon')
ax[1].set_title('Model F1-Score Comparison')
ax[1].set_ylabel('F1-Score')
ax[1].set_ylim(0, 1)
ax[1].set_xticklabels(results_df["Model"], rotation=45, ha="right")

plt.tight_layout()
plt.show()