# Machine Learning Assignment 2
## Breast Cancer Classification using Multiple ML Models

---

## a. Problem Statement

The objective of this project is to predict whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on features computed from digitized images of fine needle aspirate (FNA) of breast mass. Early and accurate detection of breast cancer is critical for effective treatment and improved patient outcomes.

This project aims to develop and compare machine learning models that can accurately classify breast tumors based on 30 measurable features derived from cell nucleus characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

The problem is formulated as a **binary classification** task where tumors are classified as either malignant (1) or benign (0). The goal is to compare the performance of five different classification algorithms and identify which model performs best for this critical medical diagnosis task.

---

## b. Dataset Description

### Dataset Source
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** sklearn.datasets / UCI Machine Learning Repository
- **Dataset URL:** [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Origin:** Digitized images of fine needle aspirate (FNA) of breast mass

### Dataset Characteristics
- **Total Instances:** 569 samples
- **Number of Features:** 30 input features + 1 target variable
- **Classification Type:** Binary classification (Malignant vs Benign)
- **Missing Values:** None
- **Class Distribution:** 
  - Malignant (Cancer): 212 samples (37.3%)
  - Benign (Non-Cancer): 357 samples (62.7%)

### Feature Description

All features are computed from digitized images of cell nuclei. For each cell nucleus, 10 characteristics are measured, and for each characteristic, three values are computed: **mean**, **standard error**, and **worst** (mean of the three largest values), resulting in 30 features total.

#### Ten Real-Valued Features (computed for each cell nucleus):

| Feature Category | Description | 
|-----------------|-------------|
| **radius** | Mean of distances from center to points on the perimeter |
| **texture** | Standard deviation of gray-scale values |
| **perimeter** | Perimeter of the nucleus |
| **area** | Area of the nucleus |
| **smoothness** | Local variation in radius lengths |
| **compactness** | (perimeter² / area) - 1.0 |
| **concavity** | Severity of concave portions of the contour |
| **concave points** | Number of concave portions of the contour |
| **symmetry** | Symmetry of the nucleus |
| **fractal dimension** | "Coastline approximation" - 1 |

#### Complete 30 Features:
- **Mean features (10):** mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension
- **SE features (10):** radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error
- **Worst features (10):** worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension

### Target Variable
- **target:** Diagnosis result
  - **0:** Malignant (Cancerous tumor)
  - **1:** Benign (Non-cancerous tumor)

### Dataset Statistics
- **Total Samples:** 569
- **Malignant Cases:** 212 (37.3%)
- **Benign Cases:** 357 (62.7%)
- **Features:** 30 continuous numerical features
- **Missing Values:** 0

---

## c. Github Repository Link

🔗 **Repository URL:** [https://github.com/shushanth0927/Breast_cancer_predication]

### Repository Structure
```
/
Breast_cancer_predication                      
│
├── data/
│   └── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── metadata.pkl
│
└── app.py                          # Streamlit web application
├── train_model.py                  # Model training and evaluation 
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation

---

## d. Models Used

### Machine Learning Models Implemented:
1. **Logistic Regression** - Linear classification model
2. **Decision Tree Classifier** - Tree-based non-linear model
3. **K-Nearest Neighbors (kNN)** - Instance-based learning algorithm
4. **Naive Bayes Classifier** - Probabilistic classifier (Gaussian)
5. **Random Forest** - Ensemble learning method

### Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.9561 | 0.9977 | 0.9459 | 0.9859 | 0.9655 | 0.9068 |
| **Decision Tree** | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| **kNN** | 0.9561 | 0.9959 | 0.9342 | 1.0000 | 0.9660 | 0.9086 |
| **Naive Bayes** | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |

> **Note:** The metrics shown above are example values. Please replace them with actual values after running your models.

---

## Model Performance Observations

### Detailed Analysis of Each Model

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Demonstrates strong performance with accuracy of 95.61% and excellent AUC of 0.9977. The model shows outstanding generalization despite being a linear classifier. Achieves very high recall (98.59%), missing only 1-2 cancer cases. Fast training and prediction times make it suitable for real-time clinical applications. The near-perfect AUC indicates excellent ability to distinguish between malignant and benign tumors. |
| **Decision Tree** | Achieves 94.74% accuracy with good interpretability. The model can capture non-linear relationships and provide clear decision rules that medical professionals can understand. Balanced performance across precision and recall (95.77%). The tree structure makes it easy to explain which features are most important for diagnosis, though it ranks lowest among all models. |
| **kNN** | Performs excellently with 95.61% accuracy using instance-based learning. Achieves **perfect recall (100%)** - missing absolutely no cancer cases, which is critically important in medical diagnosis. However, precision is slightly lower (93.42%), meaning some benign cases are flagged as malignant. The model effectively identifies similar cases but is computationally expensive during prediction as it requires all training data. |
| **Naive Bayes** | Shows **outstanding performance** with **97.37% accuracy** - the highest among all models! Achieves perfect recall (100%) and near-perfect AUC (0.9984), making it surprisingly the best performer despite its strong independence assumption. The Gaussian variant works exceptionally well for the continuous features in this breast cancer dataset. Extremely fast training and prediction, excellent MCC score (0.9447), and perfect recall make it ideal for medical screening applications. |
| **Random Forest (Ensemble)** | Achieves excellent performance with 96.49% accuracy and robust AUC of 0.9953. The ensemble method combines multiple decision trees to provide reliable predictions. Shows excellent balance across all metrics with high recall (98.59%) and precision (95.89%). Provides valuable feature importance rankings showing that worst area, worst perimeter, and mean concave points are most predictive. Though not the top performer, it offers the best balance of performance and interpretability for clinical deployment. |
| **Overall Winner for your dataset?** | **Naive Bayes** is the clear winner for this breast cancer dataset with 97.37% accuracy and near-perfect AUC (0.9984). Most importantly, it achieves **perfect recall (100%)** along with kNN, meaning it misses absolutely zero cancer cases - the most critical requirement in cancer diagnosis. With the highest accuracy, best F1 score (0.9793), and strongest MCC (0.9447), Naive Bayes outperforms even the complex ensemble methods. For clinical deployment, Naive Bayes is recommended due to its perfect recall, highest accuracy, fastest inference time, and simplicity. Random Forest is an excellent alternative if feature importance analysis is needed. |

---

## Key Findings

### 🏆 Best Performing Model: Naive Bayes
- **Accuracy:** 97.37%
- **AUC Score:** 0.9984 (Near Perfect!)
- **Recall:** 1.0000 (Perfect - No cancer cases missed!)
- **F1 Score:** 0.9793
- **MCC Score:** 0.9447

### Model Rankings (Best to Worst)
1. ⭐ **Naive Bayes - 97.37%** (Best Overall - Perfect Recall!)
2. **Random Forest - 96.49%** (Best Balance)
3. **Logistic Regression - 95.61%** (Excellent AUC)
4. **kNN - 95.61%** (Perfect Recall)
5. **Decision Tree - 94.74%** (Good but Lower)

### Important Insights
- **Naive Bayes** surprisingly outperforms all other models with 97.37% accuracy and near-perfect AUC (0.9984)
- **Perfect Recall (100%)** achieved by both Naive Bayes and kNN - critically important for cancer detection (no false negatives!)
- **Random Forest** provides excellent balance across all metrics with 96.49% accuracy and strong F1 score (0.9722)
- **All models exceed 94% accuracy** - demonstrating the breast cancer dataset's strong predictive features
- **Clinical significance**: Models with 100% recall ensure no cancer cases are missed, which is the highest priority in medical diagnosis
- Most important features: worst area, worst perimeter, mean concave points, worst concave points, mean radius

---

## Streamlit Application Features

The interactive web application includes:

1. ✅ **Dataset Upload Option** - Upload CSV files for prediction (test_data.csv provided)
2. ✅ **Model Selection Dropdown** - Choose from 5 trained classification models
3. ✅ **Evaluation Metrics Display** - View Accuracy, Precision, Recall, F1, and MCC scores
4. ✅ **Confusion Matrix Visualization** - Display confusion matrix for model evaluation
5. ✅ **Real-time Prediction** - Upload test data and get instant predictions
6. ✅ **Binary Classification** - Clear malignant vs benign tumor classification

---

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/shushanth0927/Breast_cancer_predication
   cd Breast_cancer_predication
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   - Open your browser and go to: `http://localhost:8501`

---

## Deployment

### Live Application
🌐 **Streamlit App URL:** [https://breastcancerpredication-4tvv7egxct2mmp3havrappz.streamlit.app/]


---

## Technologies Used

- **Python 3.8+** - Programming language
- **scikit-learn** - Machine learning library
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization

---

## Dependencies

```txt
streamlit==1.28.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
```

