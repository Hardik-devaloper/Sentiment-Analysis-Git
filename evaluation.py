"""
Phase 2 - Model Evaluation
Sentiment Analysis using:
Decision Tree + Naive Bayes + XGBoost + LSTM + Ensemble

Uses the ORIGINAL test dataset:
data/test.csv

This file does not modify engine.py or app.py.
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import engine


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join("data", "test.csv")

EMOTION_NAMES = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
}

CLASS_IDS = list(EMOTION_NAMES.keys())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def emotion_name(class_id):
    """Convert numeric class to emotion name."""
    try:
        return EMOTION_NAMES[int(class_id)]
    except Exception:
        return "Unknown"


def safe_class_id(value):
    """
    Convert a model prediction into the numeric emotion class.

    Accepts:
    - numeric class IDs: 0, 1, 2, 3, 4, 5
    - numeric strings: "0", "1", ...
    - emotion names: "Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"
    """

    if value is None:
        return -1

    # Already numeric
    try:
        numeric_value = int(value)

        if numeric_value in CLASS_IDS:
            return numeric_value
    except (ValueError, TypeError):
        pass

    # Emotion name
    if isinstance(value, str):
        normalized = value.strip().lower()

        for class_id, emotion in EMOTION_NAMES.items():
            if normalized == emotion.lower():
                return class_id

    return -1


def calculate_metrics(y_true, y_pred):
    """Calculate standard classification metrics."""

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0,
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("PHASE 2 - MODEL EVALUATION")
print("=" * 70)

print()
print("Loading test dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Test dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if "text" not in df.columns:
    raise ValueError(
        "Dataset must contain a 'text' column."
    )

if "label" not in df.columns:
    raise ValueError(
        "Dataset must contain a 'label' column."
    )


df = df.dropna(subset=["text", "label"]).copy()

df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

print(f"Dataset loaded successfully.")
print(f"Number of test samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

print()
print("Class distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# STORAGE FOR PREDICTIONS
# ============================================================

y_true = df["label"].tolist()

decision_tree_predictions = []
naive_bayes_predictions = []
xgboost_predictions = []
lstm_predictions = []

raw_ensemble_predictions = []
final_predictions = []

failed_predictions = []


# ============================================================
# RUN ALL MODELS
# ============================================================

print()
print("=" * 70)
print("RUNNING ALL MODELS")
print("=" * 70)

print()
print("This may take some time because the LSTM is evaluated too.")
print()


for index, text in enumerate(df["text"], start=1):

    try:

        result = engine.predict_with_details(text)

        # ----------------------------------------------------
        # Individual model predictions
        # ----------------------------------------------------

        dt_prediction = safe_class_id(
            result.get("decision_tree")
      )

        nb_prediction = safe_class_id(
            result.get("naive_bayes")
        )

        xgb_prediction = safe_class_id(
            result.get("xgboost")
        )

        lstm_prediction = safe_class_id(
            result.get("lstm")
        )

        # ----------------------------------------------------
        # Ensemble predictions
        # ----------------------------------------------------

        raw_ensemble = safe_class_id(
            result.get("raw_ensemble_class")
        )

        final_prediction = safe_class_id(
            result.get("final_class")
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        decision_tree_predictions.append(dt_prediction)
        naive_bayes_predictions.append(nb_prediction)
        xgboost_predictions.append(xgb_prediction)
        lstm_predictions.append(lstm_prediction)

        raw_ensemble_predictions.append(raw_ensemble)
        final_predictions.append(final_prediction)

    except Exception as error:

        print()
        print(
            f"ERROR while processing sample {index}: {error}"
        )

        failed_predictions.append(index)

        # Keep arrays aligned with y_true.
        decision_tree_predictions.append(-1)
        naive_bayes_predictions.append(-1)
        xgboost_predictions.append(-1)
        lstm_predictions.append(-1)
        raw_ensemble_predictions.append(-1)
        final_predictions.append(-1)

    # Progress information
    if index % 100 == 0 or index == len(df):

        print(
            f"Processed {index}/{len(df)} samples..."
        )


# ============================================================
# REMOVE FAILED PREDICTIONS
# ============================================================

if failed_predictions:

    print()
    print(
        f"WARNING: {len(failed_predictions)} predictions failed."
    )

    valid_indices = [
        i
        for i in range(len(y_true))
        if i + 1 not in failed_predictions
    ]

    y_true_eval = [
        y_true[i]
        for i in valid_indices
    ]

    decision_tree_eval = [
        decision_tree_predictions[i]
        for i in valid_indices
    ]

    naive_bayes_eval = [
        naive_bayes_predictions[i]
        for i in valid_indices
    ]

    xgboost_eval = [
        xgboost_predictions[i]
        for i in valid_indices
    ]

    lstm_eval = [
        lstm_predictions[i]
        for i in valid_indices
    ]

    raw_ensemble_eval = [
        raw_ensemble_predictions[i]
        for i in valid_indices
    ]

    final_eval = [
        final_predictions[i]
        for i in valid_indices
    ]

else:

    y_true_eval = y_true

    decision_tree_eval = decision_tree_predictions
    naive_bayes_eval = naive_bayes_predictions
    xgboost_eval = xgboost_predictions
    lstm_eval = lstm_predictions

    raw_ensemble_eval = raw_ensemble_predictions
    final_eval = final_predictions


# ============================================================
# MODEL PREDICTION DICTIONARY
# ============================================================

predictions = {

    "Decision Tree": decision_tree_eval,

    "Naive Bayes": naive_bayes_eval,

    "XGBoost": xgboost_eval,

    "LSTM": lstm_eval,

    "Ensemble": raw_ensemble_eval,

    "Context-Aware Ensemble": final_eval,
}


# ============================================================
# CALCULATE METRICS
# ============================================================

print()
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

results = []

for model_name, model_predictions in predictions.items():

    metrics = calculate_metrics(
        y_true_eval,
        model_predictions,
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
        }
    )


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY PERFORMANCE TABLE
# ============================================================

print()

display_df = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]:
    display_df[column] = (
        display_df[column] * 100
    ).round(2).astype(str) + "%"


print(display_df.to_string(index=False))


# ============================================================
# FIND BEST MODEL
# ============================================================

best_accuracy_model = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model",
]

best_f1_model = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model",
]

print()
print("=" * 70)
print("BEST MODELS")
print("=" * 70)

print(
    f"Best Accuracy : {best_accuracy_model}"
)

print(
    f"Best F1 Score : {best_f1_model}"
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

for model_name, model_predictions in predictions.items():

    print()
    print("=" * 70)
    print(f"{model_name.upper()} - CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true_eval,
            model_predictions,
            labels=CLASS_IDS,
            target_names=[
                EMOTION_NAMES[i]
                for i in CLASS_IDS
            ],
            zero_division=0,
        )
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

for model_name, model_predictions in predictions.items():

    cm = confusion_matrix(
        y_true_eval,
        model_predictions,
        labels=CLASS_IDS,
    )

    print()
    print(f"{model_name}:")
    print()

    cm_df = pd.DataFrame(
        cm,
        index=[
            EMOTION_NAMES[i]
            for i in CLASS_IDS
        ],
        columns=[
            EMOTION_NAMES[i]
            for i in CLASS_IDS
        ],
    )

    print(cm_df)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("evaluation_results", exist_ok=True)


# ------------------------------------------------------------
# Save performance metrics
# ------------------------------------------------------------

results_df.to_csv(
    "evaluation_results/model_metrics.csv",
    index=False,
)


# ------------------------------------------------------------
# Save all predictions
# ------------------------------------------------------------

prediction_df = pd.DataFrame(
    {
        "text": df.iloc[
            [i for i in range(len(df))
             if i + 1 not in failed_predictions]
        ]["text"].values,

        "Actual": y_true_eval,

        "Decision Tree": decision_tree_eval,

        "Naive Bayes": naive_bayes_eval,

        "XGBoost": xgboost_eval,

        "LSTM": lstm_eval,

        "Ensemble": raw_ensemble_eval,

        "Final": final_eval,
    }
)


prediction_df["Actual Emotion"] = (
    prediction_df["Actual"]
    .map(EMOTION_NAMES)
)

prediction_df["Final Emotion"] = (
    prediction_df["Final"]
    .map(EMOTION_NAMES)
)

prediction_df.to_csv(
    "evaluation_results/test_predictions.csv",
    index=False,
)


# ------------------------------------------------------------
# Save confusion matrices
# ------------------------------------------------------------

for model_name, model_predictions in predictions.items():

    cm = confusion_matrix(
        y_true_eval,
        model_predictions,
        labels=CLASS_IDS,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            EMOTION_NAMES[i]
            for i in CLASS_IDS
        ],
        columns=[
            EMOTION_NAMES[i]
            for i in CLASS_IDS
        ],
    )

    cm_df.to_csv(
        f"evaluation_results/{safe_name}_confusion_matrix.csv"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    "evaluation_results/evaluation_summary.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "SENTIMENT ANALYSIS - PHASE 2 EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test samples evaluated: {len(y_true_eval)}\n"
    )

    file.write(
        f"Failed predictions: {len(failed_predictions)}\n\n"
    )

    file.write(
        "MODEL PERFORMANCE\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    for _, row in results_df.iterrows():

        file.write(
            f"{row['Model']}\n"
        )

        file.write(
            f"  Accuracy : {row['Accuracy']:.4f}\n"
        )

        file.write(
            f"  Precision: {row['Precision']:.4f}\n"
        )

        file.write(
            f"  Recall   : {row['Recall']:.4f}\n"
        )

        file.write(
            f"  F1 Score : {row['F1 Score']:.4f}\n\n"
        )

    file.write(
        f"Best Accuracy Model: {best_accuracy_model}\n"
    )

    file.write(
        f"Best F1 Model: {best_f1_model}\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("PHASE 2 EVALUATION COMPLETED")
print("=" * 70)

print()
print("Results saved in:")

print(
    "  evaluation_results/model_metrics.csv"
)

print(
    "  evaluation_results/test_predictions.csv"
)

print(
    "  evaluation_results/evaluation_summary.txt"
)

print(
    "  evaluation_results/*_confusion_matrix.csv"
)

print()
print("Your original engine.py and app.py were not modified.")
print()