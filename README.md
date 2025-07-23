# Cancer Mutation Classification using Genomic and Clinical Text Data

This project aims to build machine learning models to classify genetic mutations into one of **nine classes** using both structured **genomic features** (`gene`, `variation`) and unstructured **clinical text**. The problem is inspired by real-world diagnostic challenges where understanding textual clinical evidence is crucial for accurate predictions.

---

## 🔍 Objective

To predict the probability of each sample (mutation) belonging to one of the nine classes, using a combination of structured genomic data and clinical text, while maintaining **interpretability** and optimizing for **multi-class log-loss**.

---

## 📁 Data Description

The dataset consists of two files:

- **training_variants / test_variants**: Contains structured information such as `ID`, `Gene`, and `Variation`.
- **training_text / test_text**: Contains unstructured clinical evidence (text) for each mutation, linked via the `ID`.

---

## ✅ Step 1: Data Preprocessing

1. **Read** gene and variation data from `training_variants.csv`.
2. **Read** clinical text data and perform the following preprocessing:
   - Load stopwords using `nltk`.
   - Replace special characters with space.
   - Normalize whitespaces and convert text to lowercase.
   - Remove stopwords and retain only meaningful tokens.
3. **Merge** both datasets using the `ID` field.
4. **Split** data into training, validation (CV), and test sets.
5. **Analyze class distribution** in train/test/validation sets (imbalanced labels).
6. **Baseline model**: Built a random classifier that outputs class probabilities summing to 1.
   - Log-loss on CV: **2.5366**
   - Log-loss on Test: **2.5016**

---

## 📊 Step 2: Univariate Feature Analysis

Performed exploratory analysis on all three input features:

- **Gene**
- **Variation**
- **Text**

Questions addressed:

- How many unique words appear in the training data?
- What are the word frequency distributions?
- How stable are features across train/test/CV?
- Is the text feature useful in predicting the target?
- How to best featurize the text field?
- What type of feature it is ?
- How many categories are there and How they are distributed?
- How to featurize this feature ?

---

## 🤖 Step 3: Model Building & Evaluation

Trained and evaluated the following machine learning models:

1. Naive Bayes  
2. k-Nearest Neighbors (KNN)  
3. Logistic Regression (with and without class balancing)  
4. Support Vector Machine (SVM)  
5. Random Forest  
6. Stacking Ensemble (Logistic Regression + SVM + NB)  
7. Majority Voting (Logistic Regression + SVM + RF)

> 📌 **CalibratedClassifierCV** was applied after each model to improve class probability estimates.

**Hyperparameter tuning** was done using **Grid Search**.

### 📈 Evaluation Metrics:
- **Multi-class Log-loss**
- **Misclassification Percentage**
- **Confusion Matrix**
- **Precision & Recall**

---

## 🏆 Results

| Model                        | Log-loss | Misclassification (%) | Notes                              |
|-----------------------------|----------|------------------------|-------------------------------------|
| Logistic Regression (balanced) | **1.10**   | **35.33%**              | ✅ Best performing model             |
| Random Forest (response coding) | 1.28     | 38.53%                 | ❌ Overfitted, worst performance     |
| Random Classifier (baseline)   | 2.50     | N/A                    | Used for marking upper bound              |

---

## 📌 Constraints

- **Interpretability** is important.
- **Class probabilities** must be output (not just hard labels).
- **Penalty on incorrect confidence** → Metric: **log-loss**.
- No constraints on latency/runtime.

---

## 🛠️ Libraries & Tools Used

- `Python`
- `scikit-learn`
- `nltk`
- `mlxtend` 
- `pandas`, `NumPy`
- `Matplotlib`, 
- `Jupyter Notebook`

---

## 📚 Summary

This project demonstrates a complete ML workflow for a challenging real-world biomedical classification task using both structured and unstructured data. Emphasis was placed on generating calibrated probabilities, evaluating using log-loss, and ensuring model interpretability.

