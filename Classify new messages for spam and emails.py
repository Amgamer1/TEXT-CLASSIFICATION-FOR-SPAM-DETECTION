import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Select dataset: 'spam' or 'email_origin'
dataset_choice = input("Choose dataset ('spam' or 'spamham'): ").strip().lower()

if dataset_choice == 'spam':
    model_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\nb_model_spam.pkl"
    vectorizer_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\vectorizer_spam.pkl"
elif dataset_choice == 'spamham':
    model_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\nb_model_spamham.pkl"
    vectorizer_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\vectorizer_spamham.pkl"
else:
    raise ValueError("Invalid dataset choice! Please select 'spam' or 'spamham'.")

# Load model and vectorizer
nb_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Messages to classify
messages = [
    "Congratulations! You've won a free iPhone. Claim now.",
    "Can we reschedule our meeting for tomorrow?",
    "Your account has been compromised. Click here to reset your password."
]

# Preprocess and classify
processed_messages = [preprocess_text(msg) for msg in messages]
messages_tfidf = vectorizer.transform(processed_messages)
predictions = nb_model.predict(messages_tfidf)

# Output predictions
print("\nClassification Results (0 = Ham, 1 = Spam):")
for msg, pred in zip(messages, predictions):
    print(f"Message: {msg}\nPrediction: {'1' if pred == 1 else '0'}\n")
