import joblib
import numpy as np

# Load the trained model and scaler
scaler_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\scaler_spambase.pkl"
model_path = r"C:\Users\Amir\PycharmProjects\pythonProject4\nb_model_spambase.pkl"

scaler = joblib.load(scaler_path)
nb_model = joblib.load(model_path)

# Input new messages in numerical feature format (57 attributes for Spambase)
new_samples = [
    [0.1, 0.28, 0.0, 0.4, 0.3, 0.0, 0.0, 0.2, 0.1, 0.0,
     0.1, 0.0, 0.2, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.3,
     0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0,
     0.0, 0.1, 0.0, 0.3, 0.1, 0.5, 0.2, 0.2, 0.1, 0.0,
     0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1,
     0.2, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0],
    [0.0, 0.1, 0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0,
     0.2, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.2,
     0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.1, 0.1, 0.2, 0.0, 0.3, 0.1, 0.1, 0.1, 0.0,
     0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
     0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
]

# Normalize the new samples
scaled_samples = scaler.transform(new_samples)

# Predict the class (0 = Legitimate, 1 = Spam)
predictions = nb_model.predict(scaled_samples)

# Output the predictions
print("\nClassification Results:")
for i, pred in enumerate(predictions):
    print(f"Message {i + 1}: {'1 (Spam)' if pred == 1 else ' 0 (Legitimate)'}")
