from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from flask import render_template
import matplotlib
from flask import render_template_string
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import sklearn
import os
from io import BytesIO
import base64
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import render_template_string
import joblib

# from xgboost import XGBooster

app = Flask(__name__)
df = pd.DataFrame()

X_train, X_test, y_train, y_test = None, None, None, None
xgb_accuracy, svm_accuracy, accuracy, fs_accuracy, ann_accuracy = (
    None,
    None,
    None,
    None,
    None,
)
precision, xgb_precision, svm_precision, fs_precision, ann_precision = (
    None,
    None,
    None,
    None,
    None,
)
recall, xgb_recall, svm_recall, fs_recall, ann_recall = None, None, None, None, None
f1, xgb_f1, svm_f1, fs_f1, ann_f1 = None, None, None, None, None


label_encoders = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global df
    if "file" not in request.files:
        return render_template("index.html", message="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", message="No selected file")

    if file:
        try:
            df = pd.read_csv(file)
            table_html = df.head().to_html()
            return render_template("index.html", table=table_html)

        except Exception as e:
            return render_template("index.html", message=f"Error: {str(e)}")


label_encoders = {}


@app.route("/preprocess")
def preprocess():
    global df, label_encoders
    if df.empty:
        return render_template(
            "index.html", message="DataFrame is empty. Upload a file first."
        )

    for column in df.columns:
        if df[column].dtype == "object":
            label_encoders[column] = sklearn.preprocessing.LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    table_html = df.head().to_html()
    return render_template(
        "index.html", table=table_html, message="Preprocessing completed."
    )


@app.route("/split")
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None and not df.empty:
        try:
            # Assume we have a dataset 'data' with 'features' as features and 'target' as the target variable
            # df.drop_duplicates(
            #     subset=[
            #         "Soil_color",
            #         "pH",
            #         "Rainfall",
            #         "Temperature",
            #         "Crop",
            #         "Fertilizer",
            #     ],
            #     inplace=True,
            # )
            df = df[
                ["Soil_color", "pH", "Rainfall", "Temperature", "Crop", "Fertilizer"]
            ]
            # X = df.drop("Fertilizer", axis=1)
            # Y = df["Fertilizer"]
            X = df[["Soil_color", "pH", "Rainfall", "Temperature", "Crop"]]
            Y = df["Fertilizer"]

            print(df["Fertilizer"].unique())
            # Create a StratifiedShuffleSplit object
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
            # Get indices for training and test rows
            for train_index, test_index in sss.split(X, Y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, Y, test_size=0.2, random_state=21, stratify=Y
            # )
            message = f"Split completed successfully. Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"

            return render_template("index.html", message=message)

        except Exception as e:
            return render_template("index.html", message=f"Error: {str(e)}")

    else:
        return render_template(
            "index.html",
            message='Error: Data not loaded or empty. Please click "Show" first.',
        )


from xgboost2 import XGBooster
import xgboost as xgb


def optimize_xgboost(X_train, y_train):
    # xgb_model = XGBooster()
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [100, 200, 300],
        "max_depth": [1, 3, 5, 7, 10],
    }
    grid_search = GridSearchCV(
        xgb_model, param_grid, scoring="accuracy", cv=3, n_jobs=3
    )
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_, grid_search.best_score_)
    return grid_search.best_params_, grid_search.best_score_


@app.route("/xgboost")
def xgboost_xgb():
    global X_train, X_test, y_train, y_test, xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_model

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template(
            "error.html", message='Data not split. Please click "Split" first.'
        )

    try:
        # best_params_, best_score_ = optimize_xgboost(X_train, y_train)
        # xgb_model = XGBooster(**best_params_)
        xgb_model = XGBooster(learning_rate=0.001, max_depth=3, n_estimators=100)
        xgb_model.fit(X_train.values, y_train.values)

        # print(best_params_, best_score_)
        # xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
        # xgb_model.fit(X_train.values, y_train.values)

        y_pred = xgb_model.predict(X_test.values)

        print("=== xgboost ===")
        print("=== y_pred ===")
        print(y_pred)
        print("=== ytest ===")
        print(y_test.values)
        xgb_accuracy = accuracy_score(y_test.values, y_pred)
        app.config["xgb_accuracy"] = xgb_accuracy

        xgb_precision = precision_score(y_test.values, y_pred, average="weighted")
        xgb_recall = recall_score(y_test.values, y_pred, average="weighted")
        xgb_f1 = f1_score(y_test.values, y_pred, average="weighted")

        print(f"XGBoost Metrics:")
        print(f"Accuracy: {xgb_accuracy:.4f}")
        print(f"Precision: {xgb_precision:.4f}")
        print(f"Recall: {xgb_recall:.4f}")
        print(f"F1-Score: {xgb_f1:.4f}")

        return render_template(
            "xgboost_result.html",
            xgb_accuracy=xgb_accuracy,
            xgb_precision=xgb_precision,
            xgb_recall=xgb_recall,
            xgb_f1=xgb_f1,
        )

    except Exception as e:
        return render_template("error.html", message=f"Error: {str(e)}")


import joblib
from svm1 import SVM


@app.route("/svm")
def svm():
    global X_train, X_test, y_train, y_test, svm_accuracy, svm_precision, svm_recall, svm_f1, svm_model

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template(
            "error.html", message='Data not split. Please click "Split" first.'
        )

    try:
        svm_model = SVM()
        svm_model.fit(X_train.values, y_train.values)

        predictions = svm_model.predict(X_test.values)

        print("=== svm ===")
        print("=== predictions ===")
        print(predictions)
        print("=== ytest ===")
        print(y_test.values)

        svm_accuracy = accuracy_score(y_test.values, predictions)

        svm_precision = precision_score(y_test.values, predictions, average="weighted")
        svm_recall = recall_score(y_test.values, predictions, average="weighted")
        svm_f1 = f1_score(y_test.values, predictions, average="weighted")

        print(f"SVM Metrics:")
        print(f"Accuracy: {svm_accuracy:.4f}")
        print(f"Precision: {svm_precision:.4f}")
        print(f"Recall: {svm_recall:.4f}")
        print(f"F1-Score: {svm_f1:.4f}")

        return render_template(
            "svm_result.html",
            svm_accuracy=svm_accuracy,
            svm_precision=svm_precision,
            svm_recall=svm_recall,
            svm_f1=svm_f1,
        )

    except Exception as e:
        return render_template("error.html", message=f"Error: {str(e)}")


from flask import request, render_template_string
from numpy import *
from ann1 import NeuralNetwork


@app.route("/ann")
def ann():
    global X_train, X_test, y_train, y_test, ann_accuracy, ann_precision, ann_recall, ann_f1, nn

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template(
            "error.html", message='Data not split. Please click "Split" first.'
        )

    try:

        inputs = np.array(X_train.values)
        outputs = np.array(y_train.values)
        nn = NeuralNetwork(inputs, outputs, learning_rate=0.01)
        for i in range(5000):  # Increase the number of training iterations
            nn.feedforward()
            nn.backprop()

        predictions = nn.predict(X_test.values)
        print(predictions)

        print("=== ANN ===")
        print("=== predictions ===")
        print(predictions)
        print("=== ytest ===")
        print(y_test.values)

        ann_accuracy = accuracy_score(y_test.values, predictions)
        ann_precision = precision_score(y_test.values, predictions, average="weighted")
        ann_recall = recall_score(y_test.values, predictions, average="weighted")
        ann_f1 = f1_score(y_test.values, predictions, average="weighted")

        print(f"ANN Metrics:")
        print(f"Accuracy: {ann_accuracy:.4f}")
        print(f"Precision: {ann_precision:.4f}")
        print(f"Recall: {ann_recall:.4f}")
        print(f"F1-Score: {ann_f1:.4f}")

        return render_template(
            "ann_result.html",
            ann_accuracy=ann_accuracy,
            ann_precision=ann_precision,
            ann_recall=ann_recall,
            ann_f1=ann_f1,
        )

    except Exception as e:
        return render_template("error.html", message=f"Error: {str(e)}")


def generate_accuracy_bar_graph():
    categories = ["XGBooster", "SVM", "ANN"]

    values = [xgb_accuracy * 100, svm_accuracy * 100, ann_accuracy * 100]

    fig, ax = plt.subplots()
    ax.plot(categories, values, color="blue")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Accuracy Comparison")

    graph_bytes = BytesIO()
    FigureCanvas(fig).print_png(graph_bytes)
    plt.close(fig)

    graph_encoded = base64.b64encode(graph_bytes.getvalue()).decode("utf-8")
    graph_html = (
        f'<img src="data:image/png;base64,{graph_encoded}" alt="Accuracy Graph">'
    )

    x1 = ["XGBooster", "SVM", "ANN"]
    precision_values = [xgb_precision * 100, svm_precision * 100, ann_precision * 100]
    fig, ax = plt.subplots()
    ax.plot(x1, precision_values, "-.", marker="o")
    ax.set_xlabel("ML Algorithms")
    ax.set_ylabel("Precision Values")
    ax.set_title("Precision Values Comparison")

    precision_bytes = BytesIO()
    FigureCanvas(fig).print_png(precision_bytes)
    plt.close(fig)

    precision_encoded = base64.b64encode(precision_bytes.getvalue()).decode("utf-8")
    precision_html = f'<img src="data:image/png;base64,{precision_encoded}" alt="Precision Values Graph">'

    recall_values = [xgb_recall * 100, svm_recall * 100, ann_recall * 100]
    fig, ax = plt.subplots()
    ax.plot(x1, recall_values, "--", marker="o", color="green")
    ax.set_xlabel("ML Algorithms")
    ax.set_ylabel("Recall Values")
    ax.set_title("Recall Values Comparison")

    recall_bytes = BytesIO()
    FigureCanvas(fig).print_png(recall_bytes)
    plt.close(fig)

    recall_encoded = base64.b64encode(recall_bytes.getvalue()).decode("utf-8")
    recall_html = (
        f'<img src="data:image/png;base64,{recall_encoded}" alt="Recall Values Graph">'
    )

    f1_values = [xgb_f1 * 100, svm_f1 * 100, ann_f1 * 100]
    fig, ax = plt.subplots()
    ax.plot(x1, f1_values, ":", marker="o", color="purple")
    ax.set_xlabel("ML Algorithms")
    ax.set_ylabel("F1-Score Values")
    ax.set_title("F1-Score Values Comparison")

    f1_bytes = BytesIO()
    FigureCanvas(fig).print_png(f1_bytes)
    plt.close(fig)

    f1_encoded = base64.b64encode(f1_bytes.getvalue()).decode("utf-8")
    f1_html = (
        f'<img src="data:image/png;base64,{f1_encoded}" alt="F1-Score Values Graph">'
    )
    return graph_html, precision_html, recall_html, f1_html


@app.route("/graph", methods=["POST"])
def generate_graph():
    graph_html, precision_plot_html, recall_plot_html, f1_plot_html = (
        generate_accuracy_bar_graph()
    )
    return render_template(
        "index.html",
        graph=graph_html,
        precision_plot=precision_plot_html,
        recall_plot=recall_plot_html,
        f1_plot=f1_plot_html,
    )


from xgboost import plot_tree


@app.route("/xgbooster_graph", methods=["POST"])
def generate_xgbooster_graph():
    global X_train, X_test, y_train, y_test, xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_model
    if request.method == "POST":
        if xgb_model is None:
            return render_template_string("index.html", xgbooster_graph=None)

        # xgb_model.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(20, 20))

        os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
        plot_tree(xgb_model, ax=ax, num_trees=0, rankdir="LR")
        plt.title("XGBooster Tree Visualization")

        graph_bytes = BytesIO()
        plt.savefig(graph_bytes, format="png")
        plt.close(fig)

        graph_encoded = base64.b64encode(graph_bytes.getvalue()).decode("utf-8")
        graph_html = f'<img src="data:image/png;base64,{graph_encoded}" alt="XGBooster Tree Visualization">'

        return render_template_string("index.html", xgbooster_graph=graph_html)


# ann_model = NeuralNet()


@app.route("/make_prediction")
def make_prediction():
    return render_template("make_prediction.html")


@app.route("/make_prediction_result", methods=["POST"])
def make_prediction_result():
    global xgb_model, X_train, y_train, label_encoders, svm_model

    if request.method == "POST":
        Soil_color = float(request.form["Soil_color"])
        pH = float(request.form["pH"])
        Rainfall = float(request.form["Rainfall"])
        Temperature = float(request.form["Temperature"])
        Crop = float(request.form["Crop"])
        # crop = float(request.form_data_parser_class)

        input_values = [[Soil_color, pH, Rainfall, Temperature, Crop]]
        print("Input values:", input_values)

        # try:
        #     svm_model
        # except (AttributeError, Exception, sklearn.exceptions.NotFittedError) as e:
        #     svm_model = SVM()

        # try:
        #     input_df = pd.DataFrame(input_values, columns=X_train.columns)
        #     svm_model.predict(input_df)
        # except (AttributeError, Exception, sklearn.exceptions.NotFittedError) as e:
        #     svm_model.fit(X_train, y_train)

        input_df = pd.DataFrame(input_values, columns=X_train.columns)

        if xgb_model:
            prediction = xgb_model.predict(input_df.values)
        elif nn:
            prediction = nn.predict(input_df.values)
        else:
            prediction = svm_model.predict(input_df.values)

        prediction_fertilizer_name = label_encoders["Fertilizer"].inverse_transform(
            prediction
        )

        print("Prediction:", prediction_fertilizer_name)

        return render_template(
            "prediction_result.html", prediction=prediction_fertilizer_name[0]
        )

    return render_template("make_prediction.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5011, debug=True)
