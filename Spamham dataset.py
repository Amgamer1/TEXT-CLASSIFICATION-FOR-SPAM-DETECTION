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
    """
    Preprocess the input text:
    - Convert to lowercase
    - Remove special characters and punctuation
    - Remove stop words
    - Apply stemming
    """
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())  # Lowercase and remove special characters
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]  # Stemming and stop word removal
    return ' '.join(tokens)

# Load dataset with correct delimiter
file_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\spamhamdata.csv"  # Ensure this matches your dataset file
try:
    data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])  # Use tab separator and rename columns
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Ensure the file path is correct.")

# Inspect the dataset's column names and first few rows
print("Dataset columns:", data.columns)
print(data.head())

# Convert labels to binary if necessary
if data['label'].dtype != int:
    data['label'] = data['label'].map({'ham': 0, 'spam': 1}).fillna(-1).astype(int)
    if data['label'].isnull().any():
        raise ValueError("Label mapping failed. Verify dataset values in the 'label' column.")

# Check class distribution
print("Class distribution:\n", data['label'].value_counts())

# Preprocess text
print("Preprocessing text data...")
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Split dataset into train-test sets with stratification
X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text with optimized parameters
vectorizer = TfidfVectorizer(max_features=1000, min_df=5)  # Adjust parameters as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train SVM model with balanced class weights
print("Training SVM model...")
svm_model = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_model.fit(X_train_tfidf, y_train)

# Save models and vectorizer
joblib.dump(nb_model, r"C:\Users\Amir\PycharmProjects\pythonProject4\nb_model_spamham.pkl")
joblib.dump(svm_model, r"C:\Users\Amir\PycharmProjects\pythonProject4\svm_model_spamham.pkl")
joblib.dump(vectorizer, r"C:\Users\Amir\PycharmProjects\pythonProject4\vectorizer_spamham.pkl")

# Evaluate Naive Bayes
y_pred_nb = nb_model.predict(X_test_tfidf)
print("\nNaive Bayes Evaluation:")
print(classification_report(y_test, y_pred_nb, zero_division=0))

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test_tfidf)
print("\nSVM Evaluation:")
print(classification_report(y_test, y_pred_svm, zero_division=0))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix (SpamHam)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix (SpamHam)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
