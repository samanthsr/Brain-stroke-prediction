# Brain-stroke-prediction
AI-powered desktop app to predict 3 types of Brain Stroke using Machine Learning and Deep Learning (Neural Network) with real MRI brain visualization.
# 🧠 Machine Learning Based Diagnosis & Prediction of Three Types of Brain Stroke

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green?logo=scikit-learn)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-lightblue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 About The Project

An **AI-powered Desktop Application** that automatically detects and classifies **three types of Brain Stroke** from patient health data using **Machine Learning and Deep Learning** models.

The system takes **10 patient health parameters** as input and within seconds predicts which type of stroke the patient has, along with a **confidence score**, **real MRI brain images** with dynamic highlighted regions, and **medical recommendations**.

---

## 🧠 Three Types of Brain Stroke Detected

| Type | Description | Key Risk Factors |
|------|-------------|-----------------|
| 🟢 **Ischemic Stroke** | Blood clot blocks blood flow to brain | High glucose, old age, hypertension |
| 🔴 **Hemorrhagic Stroke** | Blood vessel bursts inside brain | High BMI, hypertension, smoking |
| 🟡 **TIA (Mini Stroke)** | Temporary blockage that clears on its own | Moderate risk factors |

---

## 🖥️ Application Screenshots

### Patient Information Screen
> Doctor enters 10 patient health parameters

### Prediction Result Screen
> Shows stroke type, confidence score, MRI brain overlay and recommendation

---

## ✨ Features

- ✅ **4 ML Models** trained and compared
- ✅ **Real MRI Brain Images** with dynamic probability-based highlighting
- ✅ **Confidence Score** for each prediction
- ✅ **Two-panel dark UI** with sidebar navigation
- ✅ **Probability bars** for all 3 stroke types
- ✅ **Summary & Medical Recommendation** cards
- ✅ **Live date & time** display
- ✅ **About System** page
- ✅ **Patient Information** form with dropdowns

---

## 🤖 ML Models Used & Accuracy

| Model | Accuracy | Saved | Used in UI |
|-------|----------|-------|------------|
| Logistic Regression | 81.5% | ❌ | ❌ Comparison only |
| **Random Forest** | **85.5%** | ✅ | ✅ Shows in result |
| KNN | 75.4% | ❌ | ❌ Comparison only |
| **Neural Network** | **84.1%** | ✅ | ✅ Main prediction |

---

## 🗂️ Project Structure

```
brain_stroke_advanced/
│
├── stroke_advanced.py       # ML Training + Model Saving
├── stroke_ui.py             # Tkinter Desktop UI
├── requirements.txt         # Python dependencies
│
├── stroke_model.pkl         # Saved Random Forest model
├── stroke_nn_model.keras    # Saved Neural Network model
├── stroke_scaler.pkl        # Saved StandardScaler
├── stroke_labels.pkl        # Saved stroke type labels
│
├── ischemic_brain.png       # MRI image — Ischemic
├── hemorrhagic_brain.png    # MRI image — Hemorrhagic
├── tia_brain.png            # MRI image — TIA
│
├── correlation_heatmap.png  # Feature correlation chart
├── nn_training_curves.png   # NN accuracy & loss curves
├── model_comparison.png     # All model accuracy chart
├── confusion_matrix_nn.png  # Neural network confusion matrix
├── feature_importance.png   # Random Forest feature importance
│
└── dataset/
    └── brain_stroke.csv     # Kaggle Brain Stroke Dataset
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10 |
| **ML Framework** | Scikit-learn |
| **Deep Learning** | TensorFlow 2.x / Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **UI Framework** | Tkinter |
| **Image Processing** | Pillow (PIL) |
| **Model Saving** | Pickle |
| **Class Balancing** | Synthetic Data + Class Weights |
| **IDE** | VS Code |

---

## 📊 Dataset

- **Source:** [Kaggle — Brain Stroke Dataset](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset)
- **Size:** 4,981 patient records
- **Type:** Tabular / CSV
- **Features:** 10 patient health parameters

### Input Features

| Feature | Description | Type |
|---------|-------------|------|
| gender | Male / Female | Categorical |
| age | Patient age | Numerical |
| hypertension | High blood pressure (0/1) | Binary |
| heart_disease | Has heart disease (0/1) | Binary |
| ever_married | Married status (Yes/No) | Categorical |
| work_type | Type of work | Categorical |
| Residence_type | Urban / Rural | Categorical |
| avg_glucose_level | Blood sugar level (mg/dL) | Numerical |
| bmi | Body Mass Index | Numerical |
| smoking_status | Smoking habits | Categorical |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- VS Code (recommended)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/brain-stroke-prediction.git
cd brain-stroke-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download Dataset
1. Go to [Kaggle Brain Stroke Dataset](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset)
2. Download `brain_stroke.csv`
3. Place it inside the `dataset/` folder

### Step 4 — Add MRI Brain Images
Place these 3 MRI scan images in the root folder:
- `ischemic_brain.png`
- `hemorrhagic_brain.png`
- `tia_brain.png`

### Step 5 — Train the Models
```bash
python stroke_advanced.py
```
> ⚠️ Close each chart window when it opens to continue training

### Step 6 — Run the Application
```bash
python stroke_ui.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
imbalanced-learn
Pillow
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow imbalanced-learn Pillow
```

---

## 🧪 Test Patients

### 🔴 Ischemic Stroke Patient
| Field | Value |
|-------|-------|
| Gender | Male |
| Age | 72 |
| Hypertension | Yes |
| Heart Disease | Yes |
| Avg Glucose | 220.0 |
| BMI | 32.0 |
| Smoking | smokes |

### 🟠 Hemorrhagic Stroke Patient
| Field | Value |
|-------|-------|
| Gender | Female |
| Age | 58 |
| Hypertension | Yes |
| Heart Disease | No |
| Avg Glucose | 130.0 |
| BMI | 42.0 |
| Smoking | formerly smoked |

### 🟡 TIA Stroke Patient
| Field | Value |
|-------|-------|
| Gender | Male |
| Age | 48 |
| Hypertension | Yes |
| Heart Disease | Yes |
| Avg Glucose | 115.0 |
| BMI | 27.0 |
| Smoking | never smoked |

### ✅ No Stroke Patient
| Field | Value |
|-------|-------|
| Gender | Female |
| Age | 28 |
| Hypertension | No |
| Heart Disease | No |
| Avg Glucose | 85.0 |
| BMI | 22.0 |
| Smoking | never smoked |

---

## 📈 Model Evaluation

### Neural Network Training
- Trained for up to **150 epochs** with Early Stopping
- Validation accuracy: **~84%**
- Used **class weights** to handle imbalanced data

### Feature Importance (Random Forest)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | avg_glucose_level | 39% |
| 2 | bmi | 32% |
| 3 | age | 16% |
| 4 | hypertension | 5% |
| 5 | heart_disease | 3% |

---

## 🔄 How It Works

```
Patient enters 10 health parameters
            ↓
Data encoded & normalized (StandardScaler)
            ↓
        ┌───────────────────┐
        │   Neural Network  │ → probability for each stroke type
        │   Random Forest   │ → majority vote prediction
        └───────────────────┘
            ↓
Highest probability = Predicted Stroke Type
            ↓
MRI Brain Image loaded + overlay applied
(region size = based on probability %)
            ↓
Result shown: Type + Confidence + Summary + Recommendation
```

---

## ⚠️ Disclaimer

> This system is for **educational and research purposes only**.
> It is **NOT a substitute** for professional medical diagnosis.
> Always consult a qualified neurologist for medical decisions.



---

<p align="center">Made with ❤️ using Python & TensorFlow</p>
