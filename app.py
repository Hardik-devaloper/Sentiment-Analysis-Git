import streamlit as st

from engine import predict_sentiment


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# SENTIMENT NAMES
# ============================================================

SENTIMENT_NAMES = {
    0: "Sadness 😢",
    1: "Joy 😄",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😮"
}


# ============================================================
# TITLE
# ============================================================

st.title("📊 Sentiment Analysis")

st.subheader("ML + LSTM Ensemble")

st.write(
    "This application predicts sentiment using an ensemble "
    "of four machine learning models:"
)

st.write(
    "🌳 Decision Tree  •  🧠 Naive Bayes  •  "
    "🚀 XGBoost  •  🔥 LSTM"
)

st.write(
    "The final prediction is generated using ensemble voting."
)


# ============================================================
# TEXT INPUT
# ============================================================

user_text = st.text_area(
    "Enter your text here:",
    height=180,
    placeholder="Example: I love this product!"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    # Check empty input
    if not user_text.strip():

        st.warning("⚠️ Please enter some text!")

    else:

        with st.spinner("Analyzing sentiment..."):

            try:

                # Get prediction from engine
                result = predict_sentiment(user_text)

                # Convert NumPy integer to normal Python integer
                result = int(result)

                # Convert class number to sentiment name
                sentiment = SENTIMENT_NAMES.get(
                    result,
                    f"Unknown ({result})"
                )

                # Display result
                st.success(
                    f"### Predicted Sentiment: {sentiment}"
                )

                # Show model class
                st.info(
                    f"Model class: **{result}**"
                )

            except Exception as e:

                st.error(
                    "❌ An error occurred while analyzing the text."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sentiment Analysis using Decision Tree, Naive Bayes, "
    "XGBoost and LSTM Ensemble Learning"
)