import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")
