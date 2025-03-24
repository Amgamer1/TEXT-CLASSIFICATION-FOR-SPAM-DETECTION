import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Download stopwords if not already available
nltk.download('stopwords')

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Load dataset
file_path = r'C:\Users\Amir\PycharmProjects\pythonProject4\spam.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Rename columns for clarity
data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

# Convert labels to binary (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess text
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Split dataset into train-test sets
X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Save models and vectorizer
joblib.dump(nb_model, r"C:\Users\Amir\PycharmProjects\pythonProject4\nb_model_spam.pkl")
joblib.dump(svm_model, r"C:\Users\Amir\PycharmProjects\pythonProject4\svm_model_spam.pkl")
joblib.dump(vectorizer, r"C:\Users\Amir\PycharmProjects\pythonProject4\vectorizer_spam.pkl")

# Evaluate Naive Bayes
y_pred_nb = nb_model.predict(X_test_tfidf)
print("\nNaive Bayes Evaluation:")
print(classification_report(y_test, y_pred_nb))

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test_tfidf)
print("\nSVM Evaluation:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
