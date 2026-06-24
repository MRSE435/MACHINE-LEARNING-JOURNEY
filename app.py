from flask import Flask, request, jsonify,render_template
import pickle
import matplotlib.pyplot as plt
app = Flask(__name__)

# load saved model
with open("model.pkl", "rb") as file:
     saved_data = pickle.load(file)
model = saved_data["model"]
X = saved_data["X"]
y = saved_data["y"]

@app.route("/")
def home():
    return render_template("index.html")

    

@app.route("/predict", methods=["POST"])
def predict():
    

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    predicted_values = model.predict(X)
    prediction = float(model.predict([[study_hours, attendance]])[0])

    plt.figure()
    plt.scatter(X["study_hours"], y, label="Actual Data")
    plt.plot(X["study_hours"], predicted_values, label="Model Prediction Line")
    plt.scatter(study_hours, prediction, marker="x", s=100, label="Your Prediction")

    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")
    plt.title("Actual vs Predicted Score")
    plt.legend()

    graph_path = "static/graph.png"
    plt.savefig(graph_path)
    plt.close()

    return render_template(
        "index.html",
        prediction=round(float(prediction), 2),
        graph="graph.png"
    )

if __name__ == "__main__":
    app.run(debug=True)