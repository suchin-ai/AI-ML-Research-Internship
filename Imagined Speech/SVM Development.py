import h5py
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------- Load and Flatten EEG Data --------------------

file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Split_Data.h5'
with h5py.File(file_path, 'r') as f:
    X_train = np.array(f['X_train'])  # (560, 750, 7)
    X_test = np.array(f['X_test'])    # (140, 750, 7)
    y_train = np.array(f['y_train']).astype(str)
    y_test = np.array(f['y_test']).astype(str)

X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# Encode labels
unique_labels = np.unique(y)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
y_encoded = np.array([label_to_index[label] for label in y])

# Flatten EEG: (750, 7) → (5250,)
X_flat = X.reshape(X.shape[0], -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# -------------------- SVM Pipelines --------------------

# 1. Linear SVM (fast)
pipeline_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=300)),
    ('clf', LinearSVC(C=0.1, max_iter=5000))
])

pipeline_linear.fit(X_train, y_train)
y_pred_linear = pipeline_linear.predict(X_test)

# 2. Bagging SVM (ensemble of SVCs)
base_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
bagged_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=300)),
    ('clf', BaggingClassifier(estimator=base_svm, n_estimators=10, n_jobs=-1, random_state=42))
])

bagged_svm.fit(X_train, y_train)
y_pred_bag = bagged_svm.predict(X_test)

# -------------------- Evaluate Both --------------------

def evaluate(y_true, y_pred, title):
    print(f"\n🧪 Results: {title}")
    print(classification_report(y_true, y_pred, target_names=unique_labels))
    print("Kappa Score:", cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=unique_labels).plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix ({title})")
    plt.show()

# Run evaluations
evaluate(y_test, y_pred_linear, "LinearSVC")
evaluate(y_test, y_pred_bag, "Bagging SVM")
