import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Simulated dataset
data = {
    "age": np.random.randint(18, 60, 500),
    "gender": np.random.choice(["male", "female"], 500),
    "height": np.random.randint(150, 200, 500),
    "weight": np.random.randint(50, 120, 500),
    "activity_level": np.random.choice(["sedentary", "lightly_active", "active", "very_active"], 500),
    "carbs_pct": np.random.uniform(40, 60, 500),
    "protein_pct": np.random.uniform(20, 30, 500),
    "fat_pct": np.random.uniform(20, 30, 500),
}

df = pd.DataFrame(data)

# Encode categorical features
label_enc = LabelEncoder()
df["gender"] = label_enc.fit_transform(df["gender"])  # Male: 1, Female: 0
df["activity_level"] = df["activity_level"].map({"sedentary": 0, "lightly_active": 1, "active": 2, "very_active": 3})

# Features & Targets
X = df[["age", "gender", "height", "weight", "activity_level"]]
Y = df[["carbs_pct", "protein_pct", "fat_pct"]]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X_train, Y_train)

# Save Model & Scaler
joblib.dump(model, "models/nutrition_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_enc, "models/label_encoder.pkl")

print("✅ Model training complete & saved!")
