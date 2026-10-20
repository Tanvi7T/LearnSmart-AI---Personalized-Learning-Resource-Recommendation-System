import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="LearnSmart AI",
    page_icon="📚",
    layout="centered"
)

st.title("📚 LearnSmart AI")
st.subheader("Personalized Learning Resource Recommendation System")
st.write("Learn Smarter, Grow Faster")
st.write("Find learning resources based on your subject, level, and goal.")

@st.cache_data
def load_resources():
    path = Path(__file__).resolve().parent / "resources.csv"
    df = pd.read_csv(path)

    required = [
        "title", "subject", "level", "type",
        "topics", "duration", "url"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    for col in ["title", "subject", "level", "type", "topics", "url"]:
        df[col] = df[col].fillna("").astype(str)

    df["duration"] = pd.to_numeric(
        df["duration"], errors="coerce"
    ).fillna(0)

    df["content"] = (
        df["subject"] + " " +
        df["level"] + " " +
        df["type"] + " " +
        df["topics"]
    )
    return df

try:
    resources = load_resources()

    @st.cache_resource
    def build_model(content):
        model = TfidfVectorizer(stop_words="english")
        vectors = model.fit_transform(content)
        return model, vectors

    vectorizer, resource_vectors = build_model(
        tuple(resources["content"])
    )

    subjects = [
        "Python", "Machine Learning", "Web Development",
        "Computer Science", "Data Analysis"
    ]

    with st.form("recommendation_form"):
        subject = st.selectbox("Choose a subject", subjects)
        level = st.selectbox(
            "Choose your learning level",
            ["Beginner", "Intermediate", "Advanced"]
        )
        goal = st.text_input(
            "Your learning goal",
            placeholder="e.g. Understand classification algorithms"
        )
        submitted = st.form_submit_button(
            "Recommend Learning Resources",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not goal.strip():
            st.warning("Please enter your learning goal.")
        else:
            profile = f"{subject} {level} {goal.strip()}"
            student_vector = vectorizer.transform([profile])
            scores = cosine_similarity(
                student_vector, resource_vectors
            ).flatten()
            best_indices = scores.argsort()[::-1][:3]

            st.markdown("### 🎯 Your Recommended Resources")

            for rank, index in enumerate(best_indices, start=1):
                item = resources.iloc[index]

                with st.container(border=True):
                    st.markdown(f"**{rank}. {item['title']}**")
                    st.write(
                        f"**Subject:** {item['subject']}  \n"
                        f"**Level:** {item['level']}  \n"
                        f"**Type:** {item['type']}  \n"
                        f"**Duration:** {int(item['duration'])} minutes"
                    )
                    st.progress(
                        max(0.0, min(1.0, float(scores[index])))
                    )
                    st.caption(
                        f"Text similarity: {scores[index] * 100:.2f}%"
                    )

                    url = str(item["url"]).strip()
                    if url.startswith(("https://", "http://")):
                        st.link_button(
                            "Open Learning Resource", url
                        )

except Exception as error:
    st.error("The app could not load its learning resources.")
    st.caption(f"Details: {error}")
