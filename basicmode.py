import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# sample data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6],
    "attendance": [50, 60, 65, 70, 80, 90],
    "exam_score": [35, 45, 50, 60, 75, 90]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance"]]
y = df["exam_score"]

model = LinearRegression()
model.fit(X, y)

# save trained model inside pickle file
with open("model.pkl", "wb") as file:
    pickle.dump({
    "model": model,
    "X": X,
    "y": y
}, file)

print("Model saved successfully as model.pkl")