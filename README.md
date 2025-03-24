# TEXT CLASSIFICATION FOR SPAM DETECTION

Welcome to the **Text Classification for Spam Detection** repository! This project focuses on using machine learning techniques to classify text messages as spam or ham (non-spam). The repository includes datasets, Python scripts, and project reports.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Methods and Technologies](#methods-and-technologies)
4. [Project Files](#project-files)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [References](#references)

---

## Project Overview
This project aims to demonstrate the effectiveness of machine learning models in text classification, particularly for spam detection. The models were trained and evaluated on real-world datasets to predict whether a given message is spam or not.

---

## Datasets
The project uses the following datasets:
1. **SMS Spam Collection**: A collection of SMS messages tagged as spam or ham.
2. **Spam-Ham Dataset**: Contains labeled text data for spam and ham classification.
3. **Spambase Dataset**: Features extracted from email data, categorized into spam or non-spam.

These datasets are included repository

---

## Methods and Technologies
- **Languages**: Python
- **Libraries**: 
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib/Seaborn (for visualization)
- **Machine Learning Models**:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Support Vector Machines (SVM)

---

## Project Files
- **Datasets**:
  - Stored as 'Spam.csv' for SMS Spam Collection
  - Stored as 'Spamhamdata.csv' for Spam-Ham Dataset
  - Stored as 'Spambase.data' for Spambase Dataset
- **Models used**:
  - nb_model_spam.pkl
  - nb_model_spamham.pkl
  - nb_model_spambase.pkl
  - scaler_spambase.pkl
  - svm_model_spam.pkl
  - svm_model_spambase.pkl
  - svm_model_spamham.pkl
  - vectorizer_spam.pkl
  - vectorizer_spamham.pkl
- **Scripts**:
  - `Sms spam detection small dataset.py`: Implements baseline model.
  - `Spambase dataset.py`: Implements baseline model.
  - `Spamham dataset.py`: Implements baseline model.
  - `Classify new messages for spam and soamham.py`: Model testing
  - `Classify new messages for spambase dataset.py`: Model testing
    
---

## How to Run
1. Clone the repository:
   ```bash
   https://github.com/Amgamer1/TEXT-CLASSIFICATION-FOR-SPAM-DETECTION.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts:
   ```bash
   python Sms spam detection small dataset.py
   ```
   and
   
   ```bash
   python Spambase dataset.py
   ```
   and
    ```bash
   python Spamham dataset.py
   ```
   then run:
   
   ```bash
   python Classify new messages for spam and spamham.py
   ```
   and
    ```bash
   python Classify new messages for spambase dataset.py
   ```


   
---

## Results
- Models were evaluated based on metrics such as accuracy, precision, recall, and F1-score.
- Visualization charts demonstrate the performance of each model.
- Detailed results can be found in the `20250324_Amir_Hallal__4232224_DLBAINLP01_SecondAttempt.pdf`.

---

Thank you for exploring this repository!
