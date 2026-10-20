from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = Flask(__name__)

# Load learning resources from the CSV file
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "resources.csv"

resources = pd.read_csv(CSV_FILE)

# Check that the required columns exist
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

# Clean missing text values
for column in ["title", "subject", "level", "type", "topics", "url"]:
    resources[column] = resources[column].fillna("").astype(str)

resources["duration"] = pd.to_numeric(
    resources["duration"], errors="coerce"
).fillna(0)

# Prepare text for the recommendation engine
resources["content"] = (
    resources["subject"] + " " +
    resources["level"] + " " +
    resources["type"] + " " +
    resources["topics"]
)

# Convert resource descriptions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
resource_vectors = vectorizer.fit_transform(resources["content"])


@app.route("/")
def home():
    return """
    <html>
        <head><title>LearnSmart AI</title></head>
        <body style="font-family:Arial;padding:30px;">
            <h1>LearnSmart AI is running!</h1>
            <p>Your AI recommendation engine is ready.</p>
            <p>Example:
            <a href="/recommend?subject=Python&level=Beginner&goal=learn%20basics">
            Test recommendations</a></p>
        </body>
    </html>
    """


@app.route("/recommend", methods=["GET"])
def recommend():
    subject = request.args.get("subject", "Python").strip()
    level = request.args.get("level", "Beginner").strip()
    goal = request.args.get(
        "goal", "learn programming basics"
    ).strip()

    if not subject or not level or not goal:
        return jsonify({
            "error": "Subject, level, and learning goal are required."
        }), 400

    # Create a profile based on the student's preferences
    student_profile = f"{subject} {level} {goal}"

    # Compare the profile with available learning resources
    student_vector = vectorizer.transform([student_profile])
    scores = cosine_similarity(
        student_vector, resource_vectors
    ).flatten()

    # Select the three most relevant resources
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
    print("Starting LearnSmart AI...")
    print("Open http://127.0.0.1:5000 in your browser.")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )