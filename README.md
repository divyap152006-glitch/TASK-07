# TASK-07
# 🧬 Breast Cancer Classification using Support Vector Machine (SVM)

This project implements a **Support Vector Machine (SVM)** classifier on the **Breast Cancer Wisconsin dataset** to distinguish between **malignant (M)** and **benign (B)** tumors.  
It demonstrates preprocessing, model training with **linear & RBF kernels**, evaluation, and **hyperparameter tuning with GridSearchCV**.

---

## 📂 Project Structure
```
breast_cancer_svm/
│── breast-cancer.csv       # Dataset file (update the path as needed)
│── svm_breast_cancer.py    # Main Python script (this code)
│── README.md               # Project documentation
```

---

## ⚙️ Installation
Clone this repo and install required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 📊 Dataset
- The dataset should be in CSV format (`breast-cancer.csv`).
- Columns:
  - `id` (optional, dropped during preprocessing)
  - `diagnosis` → Target variable (`M = Malignant`, `B = Benign`)
  - Other numerical features (mean, se, worst values of cell nuclei)

---

## 🚀 Workflow

### **1. Data Preprocessing**
- Drop the `id` column if it exists.
- Encode `diagnosis` → `M = 1`, `B = 0`.
- Split into **train (80%)** and **test (20%)** sets.
- Apply **StandardScaler** for feature normalization.

### **2. Model Training**
- Train two SVM models:
  - **Linear Kernel**
  - **RBF Kernel**

### **3. Model Evaluation**
- Compute **accuracy**, **classification report**, and **confusion matrix**.
- Visualize confusion matrix with **Seaborn heatmap**.

### **4. Hyperparameter Tuning**
- Perform **GridSearchCV** on RBF kernel:
  - Parameters: `C`, `gamma`
  - 5-fold cross-validation
- Print the **best hyperparameters** and **tuned model accuracy**.

---

## 📈 Results (Sample Output)

### Linear Kernel
```
Accuracy: ~95%
Precision, Recall, F1-score: Displayed in classification report
```

### RBF Kernel
```
Accuracy: ~97%
Precision, Recall, F1-score: Displayed in classification report
```

### Best Parameters (Example)
```
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
```

Confusion Matrix (RBF):

![Confusion Matrix](confusion_matrix.png)

---

## ▶️ How to Run
Update the dataset path in the code:

```python
file_path = r"C:\Users\YourName\Downloads\breast-cancer.csv"
```

Then run:

```bash
python svm_breast_cancer.py
```

---

## 🔮 Future Improvements
- Try other classifiers (**Random Forest, KNN, Logistic Regression**).
- Perform **feature selection** for dimensionality reduction.
- Deploy as a **web app (Flask/Streamlit)** for medical use cases.

---

## 📜 License
This project is for **educational purposes only**.  
Do **not** use it for actual medical diagnosis.
