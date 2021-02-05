from flask import Flask, render_template
from flask import request
import numpy as np
import iris_dataset
from knn import kNN

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/", methods=["post"])
def predict():
    data, X, Xtrain, Ytrain = createdict(
        request.form["sepal_length"],
        request.form["sepal_width"],
        request.form["petal_length"],
        request.form["petal_width"],
    )
    result = kNN(Xtrain, Ytrain, X)
    return render_template(
        "predict.html",
        sl=data["sepal_length"],
        sw=data["sepal_width"],
        pl=data["petal_length"],
        pw=data["petal_width"],
        result=result,
    )


def createdict(sl, sw, pl, pw):
    Xtrain, Ytrain, _, _ = iris_dataset.load(split_train_test=0.6)
    X = [sl, sw, pl, pw]
    X = np.array([[float(i) for i in X]])

    return (
        {
            "sepal_length": float(sl),
            "sepal_width": float(sw),
            "petal_length": float(pl),
            "petal_width": float(pw),
        },
        X,
        Xtrain,
        Ytrain,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
