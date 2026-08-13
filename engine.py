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


# ============================================================
# WARNING CONTROL
# ============================================================

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
# EMOTION EMOJIS
# ============================================================

EMOTION_EMOJIS = {
    0: "😢",
    1: "😄",
    2: "❤️",
    3: "😡",
    4: "😨",
    5: "😮",
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
    def from_config(
        cls,
        config
    ):

        config = dict(config)

        # Old Keras / TensorFlow models may contain
        # parameters which newer Keras does not accept.

        config.pop(
            "time_major",
            None
        )

        config.pop(
            "input_length",
            None
        )

        return cls(
            **config
        )


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

        if (
            module == "keras.preprocessing.text"
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


def load_pickle(
    path
):

    if not os.path.exists(
        path
    ):

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
                    CompatibleLSTM,

                "Bidirectional":
                    tf.keras.layers.Bidirectional,

                "Embedding":
                    tf.keras.layers.Embedding,

                "Dense":
                    tf.keras.layers.Dense,

                "Dropout":
                    tf.keras.layers.Dropout,

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

def clean_text(
    text
):

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

    # Keep hashtag word
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
# HELPER: MATCH PATTERN
# ============================================================

def contains_pattern(
    text,
    pattern
):

    try:

        return bool(
            re.search(
                pattern,
                text
            )
        )

    except Exception:

        return False


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

    if not predictions:
        return 0

    counts = Counter(
        predictions
    )

    highest_count = max(
        counts.values()
    )

    winners = [
        emotion
        for emotion, count
        in counts.items()
        if count == highest_count
    ]

    # True majority
    if len(winners) == 1:

        return int(
            winners[0]
        )

    # Tie:
    # Return first model vote for backward compatibility.
    # The context/tie-resolution layer will handle it later.

    return int(
        predictions[0]
    )


# ============================================================
# GET VOTE COUNTS
# ============================================================

def get_vote_counts(
    predictions
):

    predictions = [
        int(p)
        for p in predictions
    ]

    counts = Counter(
        predictions
    )

    return {
        emotion: int(
            counts.get(
                emotion,
                0
            )
        )
        for emotion in range(6)
    }


# ============================================================
# SAFE LABEL CONVERSION
# ============================================================

def label_from_class(class_id):

    try:
        class_id = int(class_id)
    except Exception:
        return "Unknown"

    # Use the project's known emotion mapping.
    # This avoids depending on how the old LabelEncoder
    # serialized its class labels.
    return EMOTION_NAMES.get(
        class_id,
        "Unknown"
    )

    try:

        class_id = int(
            class_id
        )

    except Exception:

        return "Unknown"

    # Prefer saved LabelEncoder
    try:

        label = le.inverse_transform(
            [class_id]
        )[0]

        return str(
            label
        )

    except Exception:

        return EMOTION_NAMES.get(
            class_id,
            "Unknown"
        )


# ============================================================
# LABEL TO CLASS
# ============================================================

def class_from_label(
    label
):

    if label is None:
        return None

    label = str(
        label
    ).strip().lower()

    for class_id, emotion in EMOTION_NAMES.items():

        if label == emotion.lower():

            return class_id

    # Handle common alternative labels

    aliases = {

        "sad": 0,
        "sadness": 0,

        "happy": 1,
        "happiness": 1,
        "joy": 1,

        "love": 2,

        "angry": 3,
        "anger": 3,

        "fear": 4,
        "afraid": 4,

        "surprise": 5,
        "surprised": 5,

    }

    return aliases.get(
        label
    )


# ============================================================
# CONTEXT SIGNAL DETECTION
# ============================================================

def get_context_scores(
    original_text
):

    text = clean_text(
        original_text
    )

    scores = {

        "sadness": 0,
        "joy": 0,
        "love": 0,
        "anger": 0,
        "fear": 0,
        "surprise": 0,

    }

    if not text:

        return scores


    # ========================================================
    # NEGATION / CONTRAST
    # ========================================================

    negative_love_patterns = [

        r"\bi do not love\b",

        r"\bi don't love\b",

        r"\bi did not love\b",

        r"\bi didn't love\b",

        r"\bnot in love\b",

        r"\bno longer love\b",

        r"\bstopped loving\b",

    ]

    negative_joy_patterns = [

        r"\bnot happy\b",

        r"\bnot very happy\b",

        r"\bnot excited\b",

        r"\bno longer happy\b",

        r"\bnever happy\b",

    ]

    negative_positive_context = {

        "love": any(
            contains_pattern(
                text,
                pattern
            )
            for pattern in negative_love_patterns
        ),

        "joy": any(
            contains_pattern(
                text,
                pattern
            )
            for pattern in negative_joy_patterns
        ),

    }


    # ========================================================
    # SADNESS / GRIEF
    # ========================================================

    sadness_patterns = [

        # Strong grief
        (r"\bleft forever\b", 6),

        (r"\bleft me forever\b", 6),

        (r"\bsaid goodbye\b.*\bforever\b", 6),

        (r"\bgoodbye forever\b", 6),

        (r"\bnever see .* again\b", 6),

        (r"\bwill never see .* again\b", 6),

        (r"\bpassed away\b", 7),

        (r"\bhas passed away\b", 7),

        (r"\bdied\b", 7),

        (r"\bdeath of\b", 7),

        (r"\blost someone\b", 6),

        (r"\bloss of\b", 6),

        (r"\bfuneral\b", 6),

        # Strong sadness
        (r"\bheartbroken\b", 6),

        (r"\bheart broken\b", 6),

        (r"\bdevastated\b", 6),

        (r"\bgrieving\b", 6),

        (r"\bgrief\b", 6),

        (r"\bdeeply sad\b", 5),

        (r"\bextremely sad\b", 5),

        (r"\bvery sad\b", 4),

        (r"\bso sad\b", 4),

        (r"\bfeel sad\b", 3),

        (r"\bfeeling sad\b", 3),

        (r"\bfeel lonely\b", 4),

        (r"\bfeeling lonely\b", 4),

        (r"\bfeel alone\b", 3),

        (r"\bfeeling alone\b", 3),

        (r"\bmiss .* so much\b", 4),

        (r"\bmissing .* so much\b", 4),

        (r"\bmiss him\b", 3),

        (r"\bmiss her\b", 3),

        (r"\bmiss them\b", 3),

        (r"\bcrying\b", 4),

        (r"\bcried\b", 4),

        (r"\btears\b", 3),

        (r"\bsorrow\b", 5),

        (r"\bsuffering\b", 3),

        (r"\bfeel empty\b", 4),

        (r"\bfeeling empty\b", 4),

        (r"\bbroken heart\b", 5),

        # Family goodbye situations
        (r"\bmy uncle .* goodbye\b", 5),

        (r"\bmy aunt .* goodbye\b", 5),

        (r"\bmy friend .* goodbye\b", 5),

        (r"\bmy father .* goodbye\b", 5),

        (r"\bmy mother .* goodbye\b", 5),

        (r"\bmy brother .* goodbye\b", 5),

        (r"\bmy sister .* goodbye\b", 5),

        (r"\bmy family .* goodbye\b", 5),

        # Negative farewell
        (r"\bpainful goodbye\b", 6),

        (r"\bpainful loss\b", 6),

    ]

    for pattern, weight in sadness_patterns:

        if contains_pattern(
            text,
            pattern
        ):

            scores["sadness"] += weight


    # ========================================================
    # FEAR
    # ========================================================

    fear_patterns = [

        (r"\bterrified\b", 6),

        (r"\bterrifying\b", 5),

        (r"\bscared\b", 4),

        (r"\bfrightened\b", 5),

        (r"\bafraid\b", 4),

        (r"\bfear\b", 4),

        (r"\bpanic\b", 5),

        (r"\bpanicking\b", 5),

        (r"\bnervous\b", 3),

        (r"\banxious\b", 4),

        (r"\bworried\b", 3),

        (r"\bthreatened\b", 5),

        (r"\bin danger\b", 6),

        (r"\bunsafe\b", 4),

        (r"\bnightmare\b", 4),

        (r"\bscared that\b", 5),

        (r"\bafraid that\b", 5),

        (r"\bterrified that\b", 6),

        (r"\bworried that\b", 4),

        (r"\bsomething bad will happen\b", 5),

    ]

    for pattern, weight in fear_patterns:

        if contains_pattern(
            text,
            pattern
        ):

            scores["fear"] += weight


    # ========================================================
    # ANGER
    # ========================================================

    anger_patterns = [

        (r"\bfurious\b", 6),

        (r"\brage\b", 6),

        (r"\braging\b", 5),

        (r"\bangry\b", 5),

        (r"\bmad at\b", 4),

        (r"\bmad with\b", 4),

        (r"\bfrustrated\b", 4),

        (r"\bfrustrating\b", 4),

        (r"\bhate\b", 5),

        (r"\bhated\b", 5),

        (r"\bdisgusted\b", 5),

        (r"\bdisappointing\b", 3),

        (r"\bdisappointed\b", 3),

        (r"\bannoyed\b", 4),

        (r"\bannoying\b", 4),

        (r"\birritated\b", 4),

        (r"\birritating\b", 4),

        (r"\bworst\b", 3),

        (r"\bunacceptable\b", 5),

        (r"\bcan'?t stand\b", 5),

        (r"\bdo not like\b", 2),

        (r"\bdon'?t like\b", 2),

    ]

    for pattern, weight in anger_patterns:

        if contains_pattern(
            text,
            pattern
        ):

            scores["anger"] += weight


    # ========================================================
    # LOVE
    # ========================================================

    love_patterns = [

        # Direct romantic love
        (r"\bi love you\b", 7),

        (r"\blove you so much\b", 7),

        (r"\blove you forever\b", 7),

        (r"\bdeeply in love\b", 7),

        (r"\bfalling in love\b", 6),

        (r"\bin love\b", 6),

        (r"\bmy beloved\b", 6),

        (r"\bmy soulmate\b", 7),

        (r"\bromantic\b", 5),

        (r"\baffection\b", 4),

        (r"\badorable\b", 4),

        # Family love
        (r"\bi love my parents\b", 7),

        (r"\bi love my mother\b", 7),

        (r"\bi love my father\b", 7),

        (r"\bi love my mom\b", 7),

        (r"\bi love my dad\b", 7),

        (r"\bi love my family\b", 7),

        (r"\bi love my brother\b", 6),

        (r"\bi love my sister\b", 6),

        (r"\bi love my friends\b", 5),

        (r"\bi love my friend\b", 5),

        (r"\blove my parents\b", 7),

        (r"\blove my mother\b", 7),

        (r"\blove my father\b", 7),

        (r"\blove my family\b", 7),

        (r"\blove my brother\b", 6),

        (r"\blove my sister\b", 6),

        # General love
        (r"\bi really love\b", 6),

        (r"\bi truly love\b", 6),

        (r"\bi absolutely love\b", 6),

        (r"\bi deeply love\b", 6),

        (r"\bi love this\b", 5),

        (r"\bi love that\b", 5),

        (r"\bi love spending time\b", 5),

        (r"\blove spending time\b", 5),

        (r"\bwith all my heart\b", 5),

        (r"\bmore than anything in the world\b", 4),

    ]

    for pattern, weight in love_patterns:

        if contains_pattern(
            text,
            pattern
        ):

            scores["love"] += weight


    # ========================================================
    # JOY / HAPPINESS
    # ========================================================

    joy_patterns = [

        (r"\bextremely happy\b", 6),

        (r"\bvery happy\b", 5),

        (r"\bso happy\b", 5),

        (r"\bhappy today\b", 4),

        (r"\bhappiest\b", 6),

        (r"\bhappy\b", 3),

        (r"\bexcited\b", 5),

        (r"\bexciting\b", 4),

        (r"\bcelebrat\w*\b", 5),

        (r"\bcongratulations\b", 5),

        (r"\bcongratulate\b", 5),

        (r"\bwon\b", 5),

        (r"\bwinner\b", 5),

        (r"\bsuccess\b", 5),

        (r"\bsuccessful\b", 5),

        (r"\bthrilled\b", 6),

        (r"\bdelighted\b", 6),

        (r"\bjoyful\b", 6),

        (r"\bawesome\b", 4),

        (r"\bamazing\b", 4),

        (r"\bwonderful\b", 5),

        (r"\bbest day\b", 5),

        (r"\bdream job\b", 5),

        (r"\bgot my dream job\b", 7),

        (r"\bproud\b", 4),

        (r"\bgrateful\b", 4),

        (r"\bthankful\b", 4),

        (r"\bglad\b", 4),

    ]

    for pattern, weight in joy_patterns:

        if contains_pattern(
            text,
            pattern
        ):

            scores["joy"] += weight


    # ========================================================
    # SURPRISE
    # ========================================================

    surprise_patterns = [

        (r"\bwow\b", 4),

        (r"\bno way\b", 5),

        (r"\bcannot believe\b", 5),

        (r"\bcan'?t believe\b", 5),

        (r"\bcould not believe\b", 5),

        (r"\bcouldn'?t believe\b", 5),

        (r"\bnever expected\b", 5),

        (r"\bunexpected\b", 5),

        (r"\bsurprised\b", 6),

        (r"\bsurprise\b", 5),

        (r"\bshocked\b", 6),

        (r"\bshocking\b", 5),

        (r"\bunbelievable\b", 5),

        (r"\bwhat a surprise\b", 7),

        (r"\bnever saw that coming\b", 7),

        (r"\bwhat just happened\b", 5),

    ]

    for pattern, weight in surprise_patterns:

        if contains_pattern(
            text,
            pattern
        ):

            scores["surprise"] += weight


    # ========================================================
    # REMOVE NEGATED LOVE / JOY SIGNALS
    # ========================================================

    if negative_positive_context["love"]:

        scores["love"] = 0

    if negative_positive_context["joy"]:

        scores["joy"] = 0


    return scores


# ============================================================
# CONTEXT-AWARE EMOTION CORRECTION
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


    scores = get_context_scores(
        original_text
    )


    # ========================================================
    # GET BEST CONTEXT EMOTION
    # ========================================================

    score_to_class = {

        "sadness": 0,
        "joy": 1,
        "love": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,

    }

    sorted_emotions = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_emotion = (
        sorted_emotions[0][0]
    )

    best_score = (
        sorted_emotions[0][1]
    )

    second_score = (
        sorted_emotions[1][1]
    )


    # ========================================================
    # IMPORTANT SPECIAL CASES
    # ========================================================

    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    goodbye_present = bool(
        re.search(
            r"\bgoodbye\b|\bfarewell\b",
            text
        )
    )

    positive_goodbye_context = bool(
        re.search(
            r"\bsee you tomorrow\b"
            r"|\bsee you soon\b"
            r"|\bsee you later\b"
            r"|\bgraduat\w*\b"
            r"|\bwedding\b"
            r"|\bparty\b"
            r"|\bcelebrat\w*\b",
            text
        )
    )

    # Normal goodbye should NOT automatically become sadness.

    if (
        goodbye_present
        and positive_goodbye_context
        and scores["sadness"] < 5
    ):

        return model_prediction


    # --------------------------------------------------------
    # STRONG GRIEF
    # --------------------------------------------------------

    if scores["sadness"] >= 5:

        return 0


    # --------------------------------------------------------
    # STRONG FEAR
    # --------------------------------------------------------

    if scores["fear"] >= 6:

        return 4


    # --------------------------------------------------------
    # STRONG ANGER
    # --------------------------------------------------------

    if scores["anger"] >= 6:

        return 3


    # --------------------------------------------------------
    # STRONG LOVE
    # --------------------------------------------------------

    if scores["love"] >= 5:

        return 2


    # --------------------------------------------------------
    # STRONG JOY
    # --------------------------------------------------------

    if scores["joy"] >= 6:

        return 1


    # --------------------------------------------------------
    # STRONG SURPRISE
    # --------------------------------------------------------

    if scores["surprise"] >= 6:

        return 5


    # ========================================================
    # HANDLE COMPETING CONTEXT
    # ========================================================

    if best_score > 0:

        # A clearly dominant context signal.
        if (
            best_score >= 5
            and best_score >= second_score + 2
        ):

            return score_to_class[
                best_emotion
            ]


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    return int(
        model_prediction
    )


# ============================================================
# SMART TIE RESOLUTION
# ============================================================

def resolve_ensemble(
    text,
    predictions
):

    model_votes = [

        int(
            predictions[
                "decision_tree"
            ]
        ),

        int(
            predictions[
                "naive_bayes"
            ]
        ),

        int(
            predictions[
                "xgboost"
            ]
        ),

        int(
            predictions[
                "lstm"
            ]
        ),

    ]

    counts = get_vote_counts(
        model_votes
    )

    highest_count = max(
        counts.values()
    )

    winners = [

        emotion

        for emotion, count
        in counts.items()

        if count == highest_count

    ]


    # ========================================================
    # TRUE MAJORITY
    # ========================================================

    if len(winners) == 1:

        ensemble_prediction = int(
            winners[0]
        )

    else:

        # ====================================================
        # 2-2 OR OTHER TIE
        # ====================================================

        context_scores = get_context_scores(
            text
        )

        context_candidates = sorted(

            context_scores.items(),

            key=lambda item: item[1],

            reverse=True

        )


        selected = None

        for emotion_name, score in context_candidates:

            emotion_class = {

                "sadness": 0,
                "joy": 1,
                "love": 2,
                "anger": 3,
                "fear": 4,
                "surprise": 5,

            }[emotion_name]

            if (
                emotion_class in winners
                and score > 0
            ):

                selected = emotion_class
                break


        if selected is not None:

            ensemble_prediction = int(
                selected
            )

        else:

            # No strong contextual information.
            # Use model order as deterministic fallback.

            ensemble_prediction = int(
                model_votes[0]
            )


    # ========================================================
    # CONTEXT CORRECTION AFTER ENSEMBLE
    # ========================================================

    final_prediction = context_correction(

        text,

        ensemble_prediction

    )


    return (
        ensemble_prediction,
        final_prediction,
        counts
    )


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


    # ========================================================
    # TF-IDF
    # ========================================================

    tfidf_vector = tfidf.transform(
        [cleaned]
    )


    # ========================================================
    # DECISION TREE
    # ========================================================

    prediction_dt = int(

        dt.predict(
            tfidf_vector
        )[0]

    )


    # ========================================================
    # NAIVE BAYES
    # ========================================================

    prediction_nb = int(

        nb.predict(
            tfidf_vector
        )[0]

    )


    # ========================================================
    # XGBOOST
    # ========================================================

    prediction_xgb = int(

        xgb.predict(
            tfidf_vector
        )[0]

    )


    # ========================================================
    # LSTM
    # ========================================================

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


    ensemble_prediction, final_prediction, _ = (
        resolve_ensemble(
            text,
            predictions
        )
    )


    return label_from_class(
        final_prediction
    )


# ============================================================
# DETAILED PREDICTION
# ============================================================

def predict_with_details(
    text
):

    # ========================================================
    # EMPTY INPUT
    # ========================================================

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

            "raw_ensemble_class":
                None,

            "context_correction":
                False,

            "vote_counts":
                {},

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

            "raw_ensemble_class":
                None,

            "context_correction":
                False,

            "vote_counts":
                {},

        }


    # ========================================================
    # GET MODEL PREDICTIONS
    # ========================================================

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

            "raw_ensemble_class":
                None,

            "context_correction":
                False,

            "vote_counts":
                {},

        }


    # ========================================================
    # INDIVIDUAL MODEL RESULTS
    # ========================================================

    prediction_dt = int(
        predictions[
            "decision_tree"
        ]
    )

    prediction_nb = int(
        predictions[
            "naive_bayes"
        ]
    )

    prediction_xgb = int(
        predictions[
            "xgboost"
        ]
    )

    prediction_lstm = int(
        predictions[
            "lstm"
        ]
    )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    ensemble_prediction, final_prediction, vote_counts = (
        resolve_ensemble(
            text,
            predictions
        )
    )


    # ========================================================
    # CONTEXT CORRECTION STATUS
    # ========================================================

    correction_applied = (

        int(
            ensemble_prediction
        )
        !=
        int(
            final_prediction
        )

    )


    # ========================================================
    # RESULT
    # ========================================================

    return {

        # Final result
        "final":
            label_from_class(
                final_prediction
            ),

        # Individual models
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

        # Ensemble before correction
        "ensemble_class":
            int(
                final_prediction
            ),

        # Final result
        "final_class":
            int(
                final_prediction
            ),

        # Raw ensemble
        "raw_ensemble_class":
            int(
                ensemble_prediction
            ),

        # Whether context changed result
        "context_correction":
            correction_applied,

        # Vote counts
        "vote_counts":
            vote_counts,

    }


# ============================================================
# OPTIONAL DEBUG FUNCTION
# ============================================================

def debug_prediction(
    text
):

    details = predict_with_details(
        text
    )

    print()
    print(
        "================================================"
    )
    print(
        "PREDICTION DEBUG"
    )
    print(
        "================================================"
    )

    print(
        "Input:",
        text
    )

    print(
        "Decision Tree:",
        details[
            "decision_tree"
        ]
    )

    print(
        "Naive Bayes:",
        details[
            "naive_bayes"
        ]
    )

    print(
        "XGBoost:",
        details[
            "xgboost"
        ]
    )

    print(
        "LSTM:",
        details[
            "lstm"
        ]
    )

    print(
        "Raw Ensemble:",
        label_from_class(
            details[
                "raw_ensemble_class"
            ]
        )
    )

    print(
        "Final:",
        details[
            "final"
        ]
    )

    print(
        "Context Correction:",
        details[
            "context_correction"
        ]
    )

    print(
        "Vote Counts:",
        details[
            "vote_counts"
        ]
    )

    print(
        "================================================"
    )

    return details


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
    "Ensemble: Majority Voting + Tie Resolution"
)

print(
    "Context Correction: Enabled"

)

print(
    "Emotion Classes: 6"
)

print(
    "=============================================="
)