import os
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    log_loss,
    recall_score,
    precision_score,
    r2_score,
    explained_variance_score,
    confusion_matrix,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import load_diabetes, load_wine, load_iris
import matplotlib.pyplot as plt
import pickle as pkl
from itertools import product
from time import sleep
from datetime import datetime


# removing due to annoyance
import warnings
from sklearn.exceptions import ConvergenceWarning

ConvergenceWarning("ignore")
warnings.filterwarnings("error")


def run():

    models_params = {
        "LinearRegression": {
            "model": LinearRegression(),
            "parameters": {"fit_intercept": [True, False]},
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(),
            "parameters": {
                "criterion": [
                    "squared_error",
                    "absolute_error",
                    "friedman_mse",
                    "poisson",
                ],
                "max_features": ["sqrt", "log2"],
                "n_estimators": [10, 50, 100],
                "max_depth": [None, 5, 10],
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression(),
            "parameters": {
                "solver": ["lbfgs", "sag", "saga"],
                "penalty": ["l1", "l2"],
                "multi_class": ["auto", "ovr", "multinomial"],
                "C": [0.01, 0.001, 0.1, 1, 10],
                "max_iter": [50000],
            },
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor(),
            "parameters": {
                "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                "learning_rate": [10, 1, 0.1, 0.01, 0.001],
                "criterion": ["friedman_mse", "squared_error"],
                "max_features": ["sqrt", "log2"],
                "alpha": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            },
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(),
            "parameters": {
                "criterion": ["gini", "entropy", "log_loss"],
                "n_estimators": [100, 1000, 10000],
                "max_features": ["sqrt", "log2", None],
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "parameters": {
                    "n_neighbors": [5, 10, 25, 50],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
            },
        },
    }

    # get user input for dataset, prediction type and KFold splits

    dataset_name = input("Choose Dataset (iris, wine, diabetes):").lower()
    dataset = return_dataset(dataset_name)
    prediction_type = input(
        "Input prediction type (classification or regression):"
    ).lower()
    kf_splits = int(input("Input number of KFold Splits:"))
    print(
        "Valid model scores: {Regression: [mae, r2 , explained_var], Classification: [f1_score, log_loss, recall, precision, roc_auc]"
    )
    plot_score = input("Choose Score to determine best model:")

    grid_results = grid_search(
        dataset.data,
        dataset.target,
        models_params,
        plot_score,
        kf_splits=kf_splits,
        model_type=prediction_type,
    )
    plot_data(grid_results, prediction_type=prediction_type)

    # for model in models_used:
    #    plot_scores_by_model(grid_results, model, score_type=plot_score)


def limit_models_params(
    models_params, classifier_models, regression_models, target_type
):
    limited_models_params = {}
    for model_name, model_params in models_params.items():
        if target_type == "classification" and model_name in classifier_models:
            limited_models_params[model_name] = model_params
        elif target_type == "regression" and model_name in regression_models:
            limited_models_params[model_name] = model_params
    return limited_models_params


def _get_param_combinations(param_dict):
    keys = param_dict.keys()
    values = param_dict.values()
    for item in product(*values):
        yield dict(zip(keys, item))


def grid_search(
    X_data, y_data, models_params, plot_score, kf_splits, model_type="log_loss"
):
    if model_type == None:
        model_type = check_target(y_data)
    else:
        model_type = model_type
    results = {}
    classifier_models = ["RandomForestClassifier", "LogisticRegression", "KNN"]
    regression_models = [
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    ]
    new_models_params = limit_models_params(
        models_params, classifier_models, regression_models, target_type=model_type
    )
    for model_name, model_dict in new_models_params.items():
        best_score = None
        best_param_set = None
        model = model_dict["model"]
        parameters = model_dict["parameters"]
        param_combinations = _get_param_combinations(parameters)

        for param_set in param_combinations:
            model.set_params(**param_set)

            kf = KFold(n_splits=kf_splits, shuffle=True)
            for train_index, test_index in kf.split(X_data):
                X_train, X_test = X_data[train_index], X_data[test_index]
                y_train, y_test = y_data[train_index], y_data[test_index]
                if model_type == "regression":
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        if model_name not in [
                            "RandomForestRegressor",
                            "GradientBoostingRegressor",
                        ]:
                            coef = model.coef_
                            intercept = model.intercept_
                        else:
                            pass
                        score = {
                            "mae": mean_absolute_error(y_test, y_pred),
                            "r2": r2_score(y_test, y_pred),
                            "explained_var": explained_variance_score(y_test, y_pred),
                        }
                        if (
                            best_score is None
                            or (
                                plot_score == "explained_var"
                                and score[plot_score] > best_score[plot_score]
                            )
                            or (
                                plot_score in ["r2", "mae"]
                                and score[plot_score] < best_score[plot_score]
                            )
                        ):
                            best_score = score
                            best_param_set = param_set

                    except ValueError as ve:
                        continue
                elif model_type == "classification":
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        score = {
                            "f1_score": f1_score(y_test, y_pred > 0.5),
                            "log_loss": log_loss(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "roc_auc": roc_auc_score(y_test, y_pred),
                        }
                        if (
                            best_score is None
                            or score[plot_score] > best_score[plot_score]
                        ):
                            best_score = score
                            best_param_set = param_set
                        sleep(1)
                    except ValueError as ve:
                        continue
        if model_type == "regression":
            results[model_name] = [
                {
                    "parameters": best_param_set,
                    "score": best_score,
                    "score_type": plot_score,
                    "coef_": coef,
                    "intercept_": intercept,
                    "pred": y_pred,
                    "actual": y_test,
                }
            ]
        else:
            results[model_name] = [
                {
                    "parameters": best_param_set,
                    "score": best_score,
                    "score_type": plot_score,
                    "pred": y_pred,
                    "actual": y_test,
                }
            ]
        with open(os.path.join(os.getcwd(), "results.pkl"), "wb") as f:
            pkl.dump(results, f)
    return results


def predict_proba(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    return self._sigmoid(linear_model)


def plot_data(data, prediction_type):
    today = datetime.now().date()
    if prediction_type == "classification":
        for model_name, model_data in data.items():
            for model_info in model_data:
                preds = model_info["pred"]
                actuals = model_info["actual"]
                score_type = model_info["score_type"]

                cm = confusion_matrix(actuals, preds.round(), labels=[1, 2])

                fig, ax = plt.subplots()
                ax.matshow(cm, cmap=plt.cm.Blues)

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")

                plt.title(f"Confusion matrix for {model_name} using {score_type}")
                plt.xlabel("Predicted label")
                plt.ylabel("True label")
                plt.show()
    elif prediction_type == "regression":
        for model_name, model_data in data.items():
            if model_name == "LinearRegression":
                for model_info in model_data:
                    preds = model_info["pred"]
                    actuals = model_info["actual"]
                    score_type = model_info["score_type"]
                    resids = actuals - preds
                    intercept = model_info["intercept_"]
                    coef = model_info["coef_"]
                    plt.hist(
                        resids,
                        bins=20,
                        color="blue",
                        label=f"{model_name.upper()} results using {score_type} for best model",
                    )
                    plt.legend()
                    plt.show()
                    # plt.savefig(f"{model_name.upper()}_hist_{today}.png")
                    plt.scatter(
                        actuals,
                        preds,
                        color="red",
                        label=f"{model_name.upper()} results using {score_type} for best model",
                    )
            # plot residual histogram for non LinerRegression Models
            else:
                for model_name, model_data in data.items():
                    for model_info in model_data:
                        preds = model_info["pred"]
                        actuals = model_info["actual"]
                        score_type = model_info["score_type"]
                        resids = actuals - preds
                        plt.hist(
                            resids,
                            bins=20,
                            color="blue",
                            label=f"{model_name.upper()} results using {score_type} for best model",
                        )
                        plt.legend()
                        plt.show()
                        # plt.savefig(f"{model_name.upper()}_hist_{today}.png")
                        plt.scatter(
                            actuals,
                            preds,
                            color="red",
                            label=f"{model_name.upper()} results using {score_type} for best model",
                        )
                        plt.legend()
                        plt.show()
                        # plt.savefig(f"{model_name.upper()}_scatter_{today}.png")


def check_target(data):
    if np.issubdtype(data.dtype, np.number):
        target_type = "regression"
    else:
        target_type = "classification"
    return target_type


def return_dataset(user_input):
    if user_input == "iris":
        dataset = load_iris()
    elif user_input == "wine":
        dataset = load_wine()
    elif user_input == "diabetes":
        dataset = load_diabetes()
    return dataset


if __name__ == "__main__":
    run()
