# ============================================================
# SENTIMENT / EMOTION ANALYSIS ENGINE
# Decision Tree + Naive Bayes + XGBoost + LSTM
# Ensemble Voting + Context-Aware Correction
# ============================================================

import os
import re
import string
import pickle
import sys
import types
import warnings
from collections import Counter

import numpy as np
import tensorflow as tf

# ------------------------------------------------------------
# Reduce unnecessary warning messages in terminal
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

ML_DIR = os.path.join(
    MODEL_DIR,
    "ml"
)

LSTM_DIR = os.path.join(
    MODEL_DIR,
    "lstm"
)

# ============================================================
# EMOTION LABELS
#
# 0 = Sadness
# 1 = Joy
# 2 = Love
# 3 = Anger
# 4 = Fear
# 5 = Surprise
# ============================================================

EMOTION_NAMES = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
}

# ============================================================
# KERAS LEGACY TOKENIZER COMPATIBILITY
# ============================================================

try:

    from keras.src.legacy.preprocessing.text import Tokenizer

    keras_preprocessing_module = types.ModuleType(
        "keras.preprocessing"
    )

    keras_text_module = types.ModuleType(
        "keras.preprocessing.text"
    )

    keras_text_module.Tokenizer = Tokenizer

    keras_preprocessing_module.text = (
        keras_text_module
    )

    sys.modules[
        "keras.preprocessing"
    ] = keras_preprocessing_module

    sys.modules[
        "keras.preprocessing.text"
    ] = keras_text_module

except Exception:
    pass


# ============================================================
# KERAS OLD LSTM COMPATIBILITY
# ============================================================

try:

    from tensorflow.keras.layers import LSTM as OriginalLSTM

except Exception:

    from keras.layers import LSTM as OriginalLSTM


@tf.keras.utils.register_keras_serializable(
    package="Compatibility"
)
class CompatibleLSTM(OriginalLSTM):

    @classmethod
    def from_config(cls, config):

        config = dict(config)

        # Old Keras models sometimes contain this
        # parameter. New Keras does not accept it.
        config.pop(
            "time_major",
            None
        )

        # Some older models can contain this.
        config.pop(
            "input_length",
            None
        )

        return cls(**config)


# ============================================================
# COMPATIBLE PICKLE LOADER
# ============================================================

class CompatibleUnpickler(
    pickle.Unpickler
):

    def find_class(
        self,
        module,
        name
    ):

        # Old Keras tokenizer
        if (
            module
            == "keras.preprocessing.text"
            and name == "Tokenizer"
        ):

            try:

                from keras.src.legacy.preprocessing.text import (
                    Tokenizer
                )

                return Tokenizer

            except Exception:
                pass

        return super().find_class(
            module,
            name
        )


def load_pickle(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found:\n{path}"
        )

    with open(
        path,
        "rb"
    ) as file:

        return CompatibleUnpickler(
            file
        ).load()


# ============================================================
# LOAD TF-IDF
# ============================================================

print(
    "Loading TF-IDF vectorizer..."
)

tfidf_path = os.path.join(
    ML_DIR,
    "tfidf_vectorizer.pkl"
)

tfidf = load_pickle(
    tfidf_path
)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print(
    "Loading label encoder..."
)

label_encoder_path = os.path.join(
    ML_DIR,
    "label_encoder.pkl"
)

le = load_pickle(
    label_encoder_path
)


# ============================================================
# LOAD DECISION TREE
# ============================================================

print(
    "Loading Decision Tree..."
)

decision_tree_path = os.path.join(
    ML_DIR,
    "decision_tree.pkl"
)

dt = load_pickle(
    decision_tree_path
)


# ============================================================
# LOAD NAIVE BAYES
# ============================================================

print(
    "Loading Naive Bayes..."
)

naive_bayes_path = os.path.join(
    ML_DIR,
    "naive_bayes.pkl"
)

nb = load_pickle(
    naive_bayes_path
)


# ============================================================
# LOAD XGBOOST
# ============================================================

print(
    "Loading XGBoost..."
)

xgboost_path = os.path.join(
    ML_DIR,
    "xgboost.pkl"
)

xgb = load_pickle(
    xgboost_path
)


# ============================================================
# LOAD LSTM TOKENIZER
# ============================================================

print(
    "Loading LSTM tokenizer..."
)

tokenizer_path = os.path.join(
    LSTM_DIR,
    "tokenizer.pkl"
)

tokenizer = load_pickle(
    tokenizer_path
)


# ============================================================
# LOAD LSTM MODEL
# ============================================================

print(
    "Loading LSTM model..."
)

lstm_model_path = os.path.join(
    LSTM_DIR,
    "lstm_model.h5"
)


# Custom objects required by the old H5 model
custom_objects = {

    "LSTM":
        CompatibleLSTM,

    "CompatibleLSTM":
        CompatibleLSTM,

    "Bidirectional":
        tf.keras.layers.Bidirectional,

    "Embedding":
        tf.keras.layers.Embedding,

    "Dense":
        tf.keras.layers.Dense,

    "Dropout":
        tf.keras.layers.Dropout,
}


try:

    lstm_model = tf.keras.models.load_model(

        lstm_model_path,

        custom_objects=custom_objects,

        compile=False

    )

except Exception as first_error:

    print(
        "Standard LSTM loading failed."
    )

    print(
        "Trying legacy compatibility mode..."
    )

    try:

        lstm_model = tf.keras.models.load_model(

            lstm_model_path,

            custom_objects={
                "LSTM":
                    CompatibleLSTM,
                "CompatibleLSTM":
                    CompatibleLSTM
            },

            compile=False

        )

    except Exception as second_error:

        raise RuntimeError(
            "\nLSTM MODEL COULD NOT BE LOADED.\n\n"
            f"First error:\n{first_error}\n\n"
            f"Second error:\n{second_error}"
        )


print(
    "LSTM model loaded successfully!"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(
        text,
        str
    ):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove hashtags but keep the word
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_vote(
    predictions
):

    predictions = [
        int(p)
        for p in predictions
    ]

    counts = Counter(
        predictions
    )

    return counts.most_common(1)[0][0]


# ============================================================
# SAFE LABEL CONVERSION
# ============================================================

def label_from_class(
    class_id
):

    class_id = int(class_id)

    # First use the saved LabelEncoder.
    try:

        label = le.inverse_transform(
            [class_id]
        )[0]

        return str(label)

    except Exception:

        # Fallback mapping
        return EMOTION_NAMES.get(
            class_id,
            "Unknown"
        )


# ============================================================
# CONTEXT-AWARE EMOTION CORRECTION
# ============================================================
#
# This does NOT replace the ML models.
#
# It only corrects very strong contextual expressions that
# TF-IDF classifiers can easily misunderstand.
#
# Priority:
#
# 1. Strong sadness / grief
# 2. Strong fear
# 3. Strong anger
# 4. Strong love
# 5. Strong joy
# 6. Strong surprise
#
# Otherwise the original ensemble prediction is retained.
# ============================================================

def context_correction(
    original_text,
    model_prediction
):

    text = clean_text(
        original_text
    )

    if not text:

        return model_prediction

    # --------------------------------------------------------
    # SADNESS / GRIEF
    # --------------------------------------------------------

    sadness_strong = [

        r"\bleft forever\b",

        r"\bleft me forever\b",

        r"\bsaid goodbye\b.*\bforever\b",

        r"\bgoodbye forever\b",

        r"\bnever see .* again\b",

        r"\bwill never see .* again\b",

        r"\bmiss .* so much\b",

        r"\bmissing .* so much\b",

        r"\bheartbroken\b",

        r"\bheart broken\b",

        r"\bdevastated\b",

        r"\bgrieving\b",

        r"\bgrief\b",

        r"\bdeeply sad\b",

        r"\bextremely sad\b",

        r"\bvery sad\b",

        r"\bso sad\b",

        r"\bfeel sad\b",

        r"\bfeeling sad\b",

        r"\bfeel lonely\b",

        r"\bfeeling lonely\b",

        r"\blost someone\b",

        r"\bloss of\b",

        r"\bdied\b",

        r"\bpassed away\b",

        r"\bfuneral\b",

        r"\btears\b",

        r"\bcrying\b",

        r"\bcried\b",

        r"\bcry\b",

        r"\bsorrow\b",

        r"\bsuffering\b",

        r"\bpainful goodbye\b",

        r"\bpainful loss\b",

        r"\bfeel empty\b",

        r"\bfeeling empty\b",

        r"\bbroken heart\b",

        r"\bmy uncle .* goodbye\b",

        r"\bmy aunt .* goodbye\b",

        r"\bmy friend .* goodbye\b",

        r"\bmy father .* goodbye\b",

        r"\bmy mother .* goodbye\b",

        r"\bmy brother .* goodbye\b",

        r"\bmy sister .* goodbye\b",

    ]

    sadness_score = 0

    for pattern in sadness_strong:

        if re.search(
            pattern,
            text
        ):

            sadness_score += 1


    # --------------------------------------------------------
    # FEAR
    # --------------------------------------------------------

    fear_words = [

        r"\bterrified\b",

        r"\bterrifying\b",

        r"\bscared\b",

        r"\bfrightened\b",

        r"\bafraid\b",

        r"\bfear\b",

        r"\bpanic\b",

        r"\bpanicking\b",

        r"\bnervous\b",

        r"\banxious\b",

        r"\bworried\b",

        r"\bthreatened\b",

        r"\bin danger\b",

        r"\bunsafe\b",

        r"\bnightmare\b",

    ]

    fear_score = 0

    for pattern in fear_words:

        if re.search(
            pattern,
            text
        ):

            fear_score += 1


    # --------------------------------------------------------
    # ANGER
    # --------------------------------------------------------

    anger_words = [

        r"\bfurious\b",

        r"\brage\b",

        r"\braging\b",

        r"\bangry\b",

        r"\bmad\b",

        r"\bfrustrated\b",

        r"\bfrustrating\b",

        r"\bhate\b",

        r"\bhated\b",

        r"\bdisgusted\b",

        r"\bdisappointing\b",

        r"\bdisappointed\b",

        r"\bannoyed\b",

        r"\bannoying\b",

        r"\birritated\b",

        r"\birritating\b",

        r"\bworst\b",

        r"\bunacceptable\b",

    ]

    anger_score = 0

    for pattern in anger_words:

        if re.search(
            pattern,
            text
        ):

            anger_score += 1


    # --------------------------------------------------------
    # LOVE
    # --------------------------------------------------------

    love_words = [

        r"\bi love you\b",

        r"\blove you so much\b",

        r"\blove you forever\b",

        r"\bdeeply in love\b",

        r"\bfalling in love\b",

        r"\bin love\b",

        r"\bmy beloved\b",

        r"\bmy soulmate\b",

        r"\bromantic\b",

        r"\baffection\b",

        r"\badorable\b",

    ]

    love_score = 0

    for pattern in love_words:

        if re.search(
            pattern,
            text
        ):

            love_score += 1


    # --------------------------------------------------------
    # JOY
    # --------------------------------------------------------

    joy_words = [

        r"\bextremely happy\b",

        r"\bvery happy\b",

        r"\bso happy\b",

        r"\bhappy today\b",

        r"\bhappiest\b",

        r"\bexcited\b",

        r"\bexciting\b",

        r"\bcelebrat(e|ing|ed|ion)\b",

        r"\bcongratulations\b",

        r"\bcongratulate\b",

        r"\bwon\b",

        r"\bwinner\b",

        r"\bsuccess\b",

        r"\bsuccessful\b",

        r"\bthrilled\b",

        r"\bdelighted\b",

        r"\bjoyful\b",

        r"\bawesome\b",

        r"\bamazing\b",

        r"\bwonderful\b",

        r"\bbest day\b",

    ]

    joy_score = 0

    for pattern in joy_words:

        if re.search(
            pattern,
            text
        ):

            joy_score += 1


    # --------------------------------------------------------
    # SURPRISE
    # --------------------------------------------------------

    surprise_words = [

        r"\bwow\b",

        r"\bno way\b",

        r"\bcannot believe\b",

        r"\bcan't believe\b",

        r"\bcould not believe\b",

        r"\bcouldn't believe\b",

        r"\bnever expected\b",

        r"\bunexpected\b",

        r"\bsurprised\b",

        r"\bsurprise\b",

        r"\bshocked\b",

        r"\bshocking\b",

        r"\bunbelievable\b",

        r"\bwhat a surprise\b",

        r"\bnever saw that coming\b",

    ]

    surprise_score = 0

    for pattern in surprise_words:

        if re.search(
            pattern,
            text
        ):

            surprise_score += 1


    # ========================================================
    # IMPORTANT CONTEXT RULES
    # ========================================================

    # A goodbye by itself does NOT automatically mean sadness.
    #
    # But goodbye + forever / loss / crying / heartbroken etc.
    # is a strong sadness signal.

    goodbye_present = bool(
        re.search(
            r"\bgoodbye\b|\bfarewell\b",
            text
        )
    )

    positive_context = bool(
        re.search(
            r"\bcelebrat\w*\b"
            r"|\bgraduat\w*\b"
            r"|\bparty\b"
            r"|\bwedding\b"
            r"|\bhappy\b",
            text
        )
    )

    # Example:
    #
    # "My uncle said goodbye to me and left forever."
    #
    # sadness_score > 0
    # goodbye_present = True
    # therefore sadness wins.

    if sadness_score >= 1:

        return 0


    # Strong fear should override a weak generic prediction.

    if fear_score >= 2:

        return 4


    # Strong anger should override weak generic prediction.

    if anger_score >= 2:

        return 3


    # Strong love.

    if love_score >= 1:

        # If there is very strong sadness as well,
        # sadness was already handled above.
        return 2


    # Strong joy.

    if joy_score >= 2:

        return 1


    # Surprise should be considered when the sentence
    # contains clear surprise language and does not contain
    # stronger negative emotion.

    if surprise_score >= 2:

        if (
            anger_score == 0
            and sadness_score == 0
            and fear_score == 0
        ):

            return 5


    # --------------------------------------------------------
    # GOODBYE WITHOUT NEGATIVE CONTEXT
    # --------------------------------------------------------
    #
    # Do NOT automatically classify:
    #
    # "Goodbye everyone, see you tomorrow"
    #
    # as sadness.
    #
    # Let the ML ensemble decide.

    if goodbye_present:

        return model_prediction


    # Otherwise preserve the actual ensemble result.

    return model_prediction


# ============================================================
# RUN ALL FOUR MODELS
# ============================================================

def get_model_predictions(
    text
):

    cleaned = clean_text(
        text
    )

    if not cleaned:

        return None


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf_vector = tfidf.transform(
        [cleaned]
    )


    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    prediction_dt = int(
        dt.predict(
            tfidf_vector
        )[0]
    )


    # --------------------------------------------------------
    # NAIVE BAYES
    # --------------------------------------------------------

    prediction_nb = int(
        nb.predict(
            tfidf_vector
        )[0]
    )


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    prediction_xgb = int(
        xgb.predict(
            tfidf_vector
        )[0]
    )


    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    sequences = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded_sequences = (
        tf.keras.preprocessing
        .sequence
        .pad_sequences(
            sequences,
            maxlen=120,
            padding="pre",
            truncating="pre"
        )
    )

    lstm_output = lstm_model.predict(
        padded_sequences,
        verbose=0
    )

    prediction_lstm = int(
        np.argmax(
            lstm_output[0]
        )
    )


    return {

        "decision_tree":
            prediction_dt,

        "naive_bayes":
            prediction_nb,

        "xgboost":
            prediction_xgb,

        "lstm":
            prediction_lstm,
    }


# ============================================================
# MAIN SENTIMENT / EMOTION FUNCTION
# ============================================================

def predict_sentiment(
    text
):

    if not isinstance(
        text,
        str
    ):

        return "Unknown"


    if not text.strip():

        return "Unknown"


    predictions = get_model_predictions(
        text
    )

    if predictions is None:

        return "Unknown"


    # --------------------------------------------------------
    # ORIGINAL ENSEMBLE
    # --------------------------------------------------------

    model_votes = [

        predictions[
            "decision_tree"
        ],

        predictions[
            "naive_bayes"
        ],

        predictions[
            "xgboost"
        ],

        predictions[
            "lstm"
        ],

    ]

    ensemble_prediction = majority_vote(
        model_votes
    )


    # --------------------------------------------------------
    # CONTEXT CORRECTION
    # --------------------------------------------------------

    final_prediction = context_correction(

        text,

        ensemble_prediction

    )


    # --------------------------------------------------------
    # RETURN LABEL
    # --------------------------------------------------------

    return label_from_class(
        final_prediction
    )


# ============================================================
# DETAILED PREDICTION
# ============================================================

def predict_with_details(
    text
):

    if not isinstance(
        text,
        str
    ):

        return {

            "final": "Unknown",

            "decision_tree":
                "Unknown",

            "naive_bayes":
                "Unknown",

            "xgboost":
                "Unknown",

            "lstm":
                "Unknown",

            "ensemble_class":
                None,

            "final_class":
                None,

        }


    if not text.strip():

        return {

            "final": "Unknown",

            "decision_tree":
                "Unknown",

            "naive_bayes":
                "Unknown",

            "xgboost":
                "Unknown",

            "lstm":
                "Unknown",

            "ensemble_class":
                None,

            "final_class":
                None,

        }


    predictions = get_model_predictions(
        text
    )


    if predictions is None:

        return {

            "final": "Unknown",

            "decision_tree":
                "Unknown",

            "naive_bayes":
                "Unknown",

            "xgboost":
                "Unknown",

            "lstm":
                "Unknown",

            "ensemble_class":
                None,

            "final_class":
                None,

        }


    # --------------------------------------------------------
    # MODEL VOTES
    # --------------------------------------------------------

    prediction_dt = predictions[
        "decision_tree"
    ]

    prediction_nb = predictions[
        "naive_bayes"
    ]

    prediction_xgb = predictions[
        "xgboost"
    ]

    prediction_lstm = predictions[
        "lstm"
    ]


    model_votes = [

        prediction_dt,

        prediction_nb,

        prediction_xgb,

        prediction_lstm,

    ]


    # --------------------------------------------------------
    # ENSEMBLE RESULT
    # --------------------------------------------------------

    ensemble_prediction = majority_vote(
        model_votes
    )


    # --------------------------------------------------------
    # CONTEXT CORRECTION
    # --------------------------------------------------------

    final_prediction = context_correction(

        text,

        ensemble_prediction

    )


    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    return {

        "final":
            label_from_class(
                final_prediction
            ),

        "decision_tree":
            label_from_class(
                prediction_dt
            ),

        "naive_bayes":
            label_from_class(
                prediction_nb
            ),

        "xgboost":
            label_from_class(
                prediction_xgb
            ),

        "lstm":
            label_from_class(
                prediction_lstm
            ),

        "ensemble_class":
            int(
                ensemble_prediction
            ),

        "final_class":
            int(
                final_prediction
            ),

    }


# ============================================================
# ENGINE STATUS
# ============================================================

print(
    "=============================================="
)

print(
    "Sentiment Analysis Engine Loaded Successfully"
)

print(
    "Models: Decision Tree + Naive Bayes + XGBoost + LSTM"
)

print(
    "Ensemble: Majority Voting"
)

print(
    "Context Correction: Enabled"
)

print(
    "=============================================="
)