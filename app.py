from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os

app = Flask(__name__)

# Load the learning resource dataset
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "resources.csv"

resources = pd.read_csv(CSV_FILE)

required_columns = [
    "title", "subject", "level", "type",
    "topics", "duration", "url"
]

missing_columns = [
    column for column in required_columns
    if column not in resources.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in resources.csv: {missing_columns}"
    )

# Clean missing values
for column in ["title", "subject", "level", "type", "topics", "url"]:
    resources[column] = resources[column].fillna("").astype(str)

resources["duration"] = pd.to_numeric(
    resources["duration"], errors="coerce"
).fillna(0)

# Prepare resource text for recommendations
resources["content"] = (
    resources["subject"] + " " +
    resources["level"] + " " +
    resources["type"] + " " +
    resources["topics"]
)

# Build the TF-IDF recommendation model
vectorizer = TfidfVectorizer(stop_words="english")
resource_vectors = vectorizer.fit_transform(resources["content"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["GET"])
def recommend():
    subject = request.args.get("subject", "Python").strip()
    level = request.args.get("level", "Beginner").strip()
    goal = request.args.get("goal", "").strip()

    if not subject or not level or not goal:
        return jsonify({
            "error": "Subject, level, and learning goal are required."
        }), 400

    student_profile = f"{subject} {level} {goal}"
    student_vector = vectorizer.transform([student_profile])

    scores = cosine_similarity(
        student_vector, resource_vectors
    ).flatten()

    best_indices = scores.argsort()[::-1][:3]
    recommendations = []

    for index in best_indices:
        item = resources.iloc[index]

        recommendations.append({
            "title": str(item["title"]),
            "subject": str(item["subject"]),
            "level": str(item["level"]),
            "type": str(item["type"]),
            "duration_minutes": int(item["duration"]),
            "url": str(item["url"]),
            "similarity_percent": round(
                float(scores[index]) * 100, 2
            )
        })

    return jsonify({
        "project": "LearnSmart AI",
        "student_profile": student_profile,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
