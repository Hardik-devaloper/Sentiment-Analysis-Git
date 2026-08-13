import streamlit as st
import pandas as pd
from datetime import datetime

from engine import predict_with_details


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Emotion Analysis AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# EMOTION DEFINITIONS
# ============================================================

EMOTIONS = {
    0: {
        "name": "Sadness",
        "emoji": "😢"
    },
    1: {
        "name": "Joy",
        "emoji": "😄"
    },
    2: {
        "name": "Love",
        "emoji": "❤️"
    },
    3: {
        "name": "Anger",
        "emoji": "😡"
    },
    4: {
        "name": "Fear",
        "emoji": "😨"
    },
    5: {
        "name": "Surprise",
        "emoji": "😮"
    }
}


# ============================================================
# REVERSE EMOTION LOOKUP
# ============================================================

EMOTION_TO_CLASS = {
    "sadness": 0,
    "sad": 0,

    "joy": 1,
    "happy": 1,

    "love": 2,

    "anger": 3,
    "angry": 3,

    "fear": 4,
    "afraid": 4,

    "surprise": 5,
    "surprised": 5
}


# ============================================================
# NORMALIZE MODEL OUTPUT
# IMPORTANT:
# Supports:
#   0
#   "0"
#   np.int64(0)
#   "Joy"
#   "😄 Joy"
# ============================================================

def normalize_class(value):

    if value is None:
        return None

    # --------------------------------------------------------
    # Already an integer
    # --------------------------------------------------------

    if isinstance(value, int):

        if value in EMOTIONS:
            return value

        return None

    # --------------------------------------------------------
    # NumPy integer / float
    # --------------------------------------------------------

    try:

        numeric_value = int(value)

        if numeric_value in EMOTIONS:
            return numeric_value

    except (TypeError, ValueError, OverflowError):
        pass

    # --------------------------------------------------------
    # String handling
    # --------------------------------------------------------

    if isinstance(value, str):

        text = value.strip().lower()

        # Direct numeric string
        try:

            numeric_value = int(text)

            if numeric_value in EMOTIONS:
                return numeric_value

        except ValueError:
            pass

        # Remove emoji if engine/UI returns:
        # "😄 Joy"
        # "❤️ Love"
        # etc.
        cleaned_text = text

        for emotion_data in EMOTIONS.values():

            cleaned_text = cleaned_text.replace(
                emotion_data["emoji"],
                ""
            )

        cleaned_text = cleaned_text.strip()

        # Direct emotion name
        if cleaned_text in EMOTION_TO_CLASS:

            return EMOTION_TO_CLASS[cleaned_text]

    return None


# ============================================================
# EMOTION DISPLAY
# ============================================================

def emotion_name(value):

    class_id = normalize_class(value)

    if class_id in EMOTIONS:

        return (
            f"{EMOTIONS[class_id]['emoji']} "
            f"{EMOTIONS[class_id]['name']}"
        )

    return "Unknown"


# ============================================================
# EMOTION NAME ONLY
# ============================================================

def emotion_name_only(value):

    class_id = normalize_class(value)

    if class_id in EMOTIONS:

        return EMOTIONS[class_id]["name"]

    return "Unknown"


# ============================================================
# EMOTION EMOJI ONLY
# ============================================================

def emotion_emoji(value):

    class_id = normalize_class(value)

    if class_id in EMOTIONS:

        return EMOTIONS[class_id]["emoji"]

    return "❓"


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "selected_text" not in st.session_state:

    st.session_state.selected_text = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Emotion Analyzer")

    st.write(
        "Analyze text using a four-model "
        "machine learning ensemble."
    )

    st.divider()

    st.subheader("🤖 Models")

    st.write("🌳 Decision Tree")
    st.write("🧠 Naive Bayes")
    st.write("🚀 XGBoost")
    st.write("🔥 LSTM")

    st.divider()

    st.subheader("🎯 Supported Emotions")

    for class_id, data in EMOTIONS.items():

        st.write(
            f"{data['emoji']} "
            f"{data['name']} — Class {class_id}"
        )

    st.divider()

    st.caption(
        "ML Project — Emotion Analysis "
        "using Ensemble Learning"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Emotion Analysis AI")

st.markdown(
    """
    ### Multi-Model Machine Learning & LSTM Ensemble

    Analyze the emotional content of text using:

    **Decision Tree • Naive Bayes • XGBoost • LSTM**

    The final prediction is selected using ensemble voting
    with the prediction engine's context-correction logic.
    """
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "Enter a sentence below. The system will run all four "
    "models and display their individual predictions "
    "along with the final ensemble prediction."
)


# ============================================================
# QUICK TESTS
# ============================================================

st.subheader("🧪 Quick Test Sentences")


quick_tests = {

    "😊 Happy":
        "I am extremely happy today because I got my dream job!",

    "😢 Sad":
        "My best friend left me forever and I feel heartbroken.",

    "😡 Angry":
        "I am extremely angry and frustrated with this terrible service.",

    "😮 Surprise":
        "Wow! I cannot believe this happened!"
}


quick_columns = st.columns(4)


for column, (label, text) in zip(
    quick_columns,
    quick_tests.items()
):

    with column:

        if st.button(
            label,
            use_container_width=True
        ):

            st.session_state.selected_text = text

            # Store selected text for rerun
            st.rerun()


# ============================================================
# TEXT INPUT
# ============================================================

st.subheader("✍️ Enter Your Text")


user_text = st.text_area(
    "Write a sentence to analyze:",
    value=st.session_state.selected_text,
    height=160,
    placeholder=(
        "Example: I love spending time with my family."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Emotion",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not user_text.strip():

        st.warning(
            "⚠️ Please enter some text before analyzing."
        )

        st.stop()


    # ========================================================
    # RUN ENGINE
    # ========================================================

    with st.spinner(
        "Running Decision Tree, Naive Bayes, "
        "XGBoost and LSTM..."
    ):

        try:

            details = predict_with_details(
                user_text.strip()
            )

        except Exception as error:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(error)

            st.stop()


    # ========================================================
    # CONVERT ALL MODEL RESULTS
    # IMPORTANT:
    # Engine can return either class IDs or emotion names.
    # normalize_class() handles both.
    # ========================================================

    decision_tree = normalize_class(
        details.get("decision_tree")
    )

    naive_bayes = normalize_class(
        details.get("naive_bayes")
    )

    xgboost = normalize_class(
        details.get("xgboost")
    )

    lstm = normalize_class(
        details.get("lstm")
    )

    ensemble_class = normalize_class(
        details.get("ensemble_class")
    )

    final_class = normalize_class(
        details.get("final_class")
    )


    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    model_predictions = {

        "Decision Tree":
            decision_tree,

        "Naive Bayes":
            naive_bayes,

        "XGBoost":
            xgboost,

        "LSTM":
            lstm
    }


    # ========================================================
    # VALID PREDICTIONS
    # ========================================================

    valid_predictions = [

        value

        for value in model_predictions.values()

        if value in EMOTIONS
    ]


    # ========================================================
    # FINAL RESULT VALIDATION
    # ========================================================

    if final_class not in EMOTIONS:

        st.error(
            "The prediction engine returned an invalid "
            f"final prediction: {details.get('final_class')}"
        )

        st.json(details)

        st.stop()


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    final_emotion = EMOTIONS[final_class]


    st.divider()

    st.subheader("🎯 Final Prediction")


    st.success(
        f"### {final_emotion['emoji']} "
        f"{final_emotion['name']}"
    )


    st.metric(
        "Final Emotion",
        final_emotion["name"]
    )


    # ========================================================
    # INDIVIDUAL MODEL PREDICTIONS
    # ========================================================

    st.subheader(
        "🤖 Individual Model Predictions"
    )


    columns = st.columns(4)


    for column, (
        model_name,
        prediction
    ) in zip(
        columns,
        model_predictions.items()
    ):

        with column:

            st.markdown(
                f"### {model_name}"
            )

            if prediction in EMOTIONS:

                st.info(
                    emotion_name(prediction)
                )

                st.caption(
                    f"Class: {prediction}"
                )

            else:

                # This should only happen if the engine
                # genuinely returns an invalid value.
                st.warning(
                    "⚠️ Unknown"
                )


    # ========================================================
    # ENSEMBLE VOTING
    # ========================================================

    st.subheader(
        "🗳️ Ensemble Voting"
    )


    vote_counts = {

        class_id: valid_predictions.count(class_id)

        for class_id in set(valid_predictions)
    }


    sorted_votes = sorted(
        vote_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )


    if sorted_votes:

        voting_columns = st.columns(
            len(sorted_votes)
        )


        for column, (
            class_id,
            votes
        ) in zip(
            voting_columns,
            sorted_votes
        ):

            with column:

                st.metric(
                    emotion_name_only(class_id),
                    f"{votes} / {len(valid_predictions)} votes"
                )


    # ========================================================
    # MODEL AGREEMENT
    # ========================================================

    if valid_predictions:

        final_vote_count = vote_counts.get(
            ensemble_class,
            0
        )

        agreement = (
            final_vote_count /
            len(valid_predictions)
        ) * 100


        st.subheader(
            "📊 Model Agreement"
        )


        st.progress(
            min(1.0, agreement / 100)
        )


        st.write(
            f"**{agreement:.0f}% of the models agreed "
            f"with the ensemble class.**"
        )


    # ========================================================
    # CONTEXT CORRECTION
    # ========================================================

    context_correction = details.get(
        "context_correction",
        False
    )


    if (
        context_correction
        and ensemble_class is not None
        and final_class != ensemble_class
    ):

        st.warning(
            "🧠 Context correction was applied by "
            "the prediction engine."
        )


        st.write(
            f"Raw ensemble prediction: "
            f"**{emotion_name(ensemble_class)}**"
        )


        st.write(
            f"Final context-aware prediction: "
            f"**{emotion_name(final_class)}**"
        )

    else:

        st.success(
            "✅ Final prediction matches the ensemble result."
        )


    # ========================================================
    # MODEL VOTING TABLE
    # ========================================================

    st.subheader(
        "📋 Model Voting Table"
    )


    voting_table = pd.DataFrame({

        "Model": [
            "Decision Tree",
            "Naive Bayes",
            "XGBoost",
            "LSTM"
        ],

        "Class": [

            decision_tree
            if decision_tree is not None
            else "Unknown",

            naive_bayes
            if naive_bayes is not None
            else "Unknown",

            xgboost
            if xgboost is not None
            else "Unknown",

            lstm
            if lstm is not None
            else "Unknown"
        ],

        "Emotion": [

            emotion_name(decision_tree),

            emotion_name(naive_bayes),

            emotion_name(xgboost),

            emotion_name(lstm)
        ]
    })


    st.dataframe(
        voting_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    history_record = {

        "Time":
            datetime.now().strftime("%H:%M:%S"),

        "Text":
            user_text.strip(),

        "Decision Tree":
            emotion_name_only(decision_tree),

        "Naive Bayes":
            emotion_name_only(naive_bayes),

        "XGBoost":
            emotion_name_only(xgboost),

        "LSTM":
            emotion_name_only(lstm),

        "Ensemble":
            emotion_name_only(ensemble_class),

        "Final":
            emotion_name_only(final_class)
    }


    st.session_state.history.append(
        history_record
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📝 Prediction History"
    )


    history_df = pd.DataFrame(
        st.session_state.history
    )


    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    csv_data = history_df.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download History as CSV",
        data=csv_data,
        file_name="emotion_prediction_history.csv",
        mime="text/csv",
        use_container_width=False
    )


    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    if st.button(
        "🗑️ Clear History"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Emotion Analysis using Decision Tree, "
    "Naive Bayes, XGBoost and LSTM Ensemble Learning"
)