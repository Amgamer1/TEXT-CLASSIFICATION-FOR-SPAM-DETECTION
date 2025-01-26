import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = r'C:\Users\Amir\PycharmProjects\pythonProject4\spambase.data'
data = pd.read_csv(file_path, header=None)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column is the label

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split dataset into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Save models and scaler
joblib.dump(nb_model, r"C:\Users\Amir\PycharmProjects\pythonProject4\nb_model_spambase.pkl")
joblib.dump(svm_model, r"C:\Users\Amir\PycharmProjects\pythonProject4\svm_model_spambase.pkl")
joblib.dump(scaler, r"C:\Users\Amir\PycharmProjects\pythonProject4\scaler_spambase.pkl")

# Evaluate Naive Bayes
y_pred_nb = nb_model.predict(X_test)
print("\nNaive Bayes Evaluation:")
print(classification_report(y_test, y_pred_nb))

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test)
print("\nSVM Evaluation:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix (Spambase)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix (Spambase)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
