# =============================================
# Train Logistic, Decision Tree, and KNN
# Compare accuracy and save best model
# =============================================

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"D:\Data_Science_and_Machine_Learning\BeansModel\BeansModel\Dry_Bean.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":
        
        
        X[col] = LabelEncoder().fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
sm = SMOTE()
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# =============================================
# MODELS TO COMPARE
# =============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


results = {}
best_accuracy = 0
best_model_name = None
best_model = None

# =============================================
# TRAINING LOOP
# =============================================

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

# =============================================
# BEST MODEL
# =============================================

print("\n===============================")
print("BEST MODEL:", best_model_name)
print("BEST ACCURACY:", best_accuracy)
print("===============================\n")

# =============================================
# SAVE (PICKLE) BEST MODEL + SCALER
# =============================================

# =============================================
# SAVE (PICKLE) BEST MODEL + SCALER + LABEL ENCODER
# =============================================

with open("bestModel.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Best model, scaler, and label encoder saved successfully!")

