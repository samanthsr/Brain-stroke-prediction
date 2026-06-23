# ============================================================
# ADVANCED Brain Stroke Prediction - ULTIMATE FIX
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("     ADVANCED BRAIN STROKE PREDICTION SYSTEM")
print("=" * 60)

# ============================================================
# STEP 1: Create Perfectly Balanced Dataset
# ============================================================

np.random.seed(42)
n = 1000  # 1000 samples per class

def make_samples(n, stroke_type):
    if stroke_type == 0:  # No Stroke
        return {
            'gender'           : np.random.randint(0, 2, n),
            'age'              : np.random.randint(20, 60, n),
            'hypertension'     : np.random.choice([0,1], n, p=[0.85,0.15]),
            'heart_disease'    : np.random.choice([0,1], n, p=[0.90,0.10]),
            'ever_married'     : np.random.randint(0, 2, n),
            'work_type'        : np.random.randint(0, 5, n),
            'Residence_type'   : np.random.randint(0, 2, n),
            'avg_glucose_level': np.random.uniform(60, 120, n),
            'bmi'              : np.random.uniform(18, 28, n),
            'smoking_status'   : np.random.randint(0, 4, n),
            'stroke_type'      : np.zeros(n, dtype=int)
        }
    elif stroke_type == 1:  # Ischemic
        return {
            'gender'           : np.random.randint(0, 2, n),
            'age'              : np.random.randint(55, 85, n),
            'hypertension'     : np.random.choice([0,1], n, p=[0.20,0.80]),
            'heart_disease'    : np.random.choice([0,1], n, p=[0.30,0.70]),
            'ever_married'     : np.random.randint(0, 2, n),
            'work_type'        : np.random.randint(0, 5, n),
            'Residence_type'   : np.random.randint(0, 2, n),
            'avg_glucose_level': np.random.uniform(150, 280, n),
            'bmi'              : np.random.uniform(26, 42, n),
            'smoking_status'   : np.random.choice([0,1,2,3], n,
                                  p=[0.10,0.30,0.35,0.25]),
            'stroke_type'      : np.ones(n, dtype=int)
        }
    elif stroke_type == 2:  # Hemorrhagic
        return {
            'gender'           : np.random.randint(0, 2, n),
            'age'              : np.random.randint(45, 80, n),
            'hypertension'     : np.random.choice([0,1], n, p=[0.15,0.85]),
            'heart_disease'    : np.random.choice([0,1], n, p=[0.40,0.60]),
            'ever_married'     : np.random.randint(0, 2, n),
            'work_type'        : np.random.randint(0, 5, n),
            'Residence_type'   : np.random.randint(0, 2, n),
            'avg_glucose_level': np.random.uniform(100, 200, n),
            'bmi'              : np.random.uniform(30, 45, n),
            'smoking_status'   : np.random.choice([0,1,2,3], n,
                                  p=[0.15,0.25,0.35,0.25]),
            'stroke_type'      : np.full(n, 2, dtype=int)
        }
    else:  # TIA
        return {
            'gender'           : np.random.randint(0, 2, n),
            'age'              : np.random.randint(40, 75, n),
            'hypertension'     : np.random.choice([0,1], n, p=[0.45,0.55]),
            'heart_disease'    : np.random.choice([0,1], n, p=[0.55,0.45]),
            'ever_married'     : np.random.randint(0, 2, n),
            'work_type'        : np.random.randint(0, 5, n),
            'Residence_type'   : np.random.randint(0, 2, n),
            'avg_glucose_level': np.random.uniform(100, 160, n),
            'bmi'              : np.random.uniform(24, 36, n),
            'smoking_status'   : np.random.choice([0,1,2,3], n,
                                  p=[0.25,0.30,0.25,0.20]),
            'stroke_type'      : np.full(n, 3, dtype=int)
        }

frames = [pd.DataFrame(make_samples(n, i)) for i in range(4)]
df = pd.concat(frames, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

stroke_labels = {0: 'No Stroke', 1: 'Ischemic',
                 2: 'Hemorrhagic', 3: 'TIA'}

print(f"\n✅ Dataset Shape: {df.shape}")
print("\n📊 Stroke Type Distribution:")
print(df['stroke_type'].value_counts().rename(stroke_labels))

# ============================================================
# STEP 2: Feature Selection & Scale
# ============================================================

feature_cols = [c for c in df.columns if c != 'stroke_type']

X = df[feature_cols]
y = df['stroke_type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2,
    random_state=42, stratify=y
)

print(f"\n✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# STEP 3: Correlation Heatmap
# ============================================================

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("\n📊 Correlation heatmap saved!")

# ============================================================
# STEP 4: Train ML Models
# ============================================================

print("\n" + "=" * 60)
print("         TRAINING ML MODELS")
print("=" * 60)

ml_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced'),
    "Random Forest"      : RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight='balanced',
        max_depth=20, min_samples_leaf=2),
    "KNN"                : KNeighborsClassifier(n_neighbors=5)
}

ml_results = {}

for name, model in ml_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    ml_results[name] = acc
    print(f"\n🔹 {name} — Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=list(stroke_labels.values())))

# ============================================================
# STEP 5: Neural Network
# ============================================================

print("\n" + "=" * 60)
print("     TRAINING DEEP LEARNING (NEURAL NETWORK)")
print("=" * 60)

num_classes  = 4
y_train_cat  = to_categorical(y_train, num_classes)
y_test_cat   = to_categorical(y_test,  num_classes)

class_weights    = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

model_nn = Sequential([
    Dense(256, activation='relu',
          input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64,  activation='relu'),
    Dropout(0.2),
    Dense(32,  activation='relu'),
    Dense(num_classes, activation='softmax')
])

model_nn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_nn.summary()

early_stop = EarlyStopping(
    monitor='val_loss', patience=15,
    restore_best_weights=True
)

print("\n🔄 Training Neural Network...")
history = model_nn.fit(
    X_train, y_train_cat,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

nn_loss, nn_acc = model_nn.evaluate(X_test, y_test_cat, verbose=0)
ml_results["Neural Network"] = nn_acc
print(f"\n✅ Neural Network Accuracy: {nn_acc*100:.2f}%")

y_pred_nn = np.argmax(model_nn.predict(X_test), axis=1)
print("\n📋 Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn,
      target_names=list(stroke_labels.values())))

# ============================================================
# STEP 6: Training Curves
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],
             label='Train Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'],
             label='Val Accuracy',   color='orange')
axes[0].set_title('Neural Network — Accuracy', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],
             label='Train Loss', color='blue')
axes[1].plot(history.history['val_loss'],
             label='Val Loss',   color='orange')
axes[1].set_title('Neural Network — Loss', fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('nn_training_curves.png')
plt.show()
print("📊 Training curves saved!")

# ============================================================
# STEP 7: Model Comparison Chart
# ============================================================

plt.figure(figsize=(10, 6))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars   = plt.bar(ml_results.keys(),
                 [v * 100 for v in ml_results.values()],
                 color=colors, edgecolor='black', width=0.5)
plt.title('Model Accuracy Comparison',
          fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 115)
for bar, val in zip(bars, ml_results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             f'{val * 100:.1f}%',
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
print("📊 Model comparison chart saved!")

# ============================================================
# STEP 8: Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(stroke_labels.values()),
            yticklabels=list(stroke_labels.values()))
plt.title('Confusion Matrix — Neural Network', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_nn.png')
plt.show()
print("📊 Confusion matrix saved!")

# ============================================================
# STEP 9: Feature Importance
# ============================================================

importances = ml_models["Random Forest"].feature_importances_
feat_df = pd.DataFrame({
    'Feature'   : feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature',
            data=feat_df, palette='viridis')
plt.title('Feature Importance — Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("📊 Feature importance chart saved!")

# ============================================================
# STEP 10: Save Models
# ============================================================

with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(ml_models["Random Forest"], f)

model_nn.save('stroke_nn_model.keras')

with open('stroke_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('stroke_labels.pkl', 'wb') as f:
    pickle.dump(stroke_labels, f)

print("\n✅ All models saved!")
print("✅ Project Training Complete!")