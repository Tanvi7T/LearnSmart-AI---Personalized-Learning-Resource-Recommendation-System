## LearnSmart AI — Personalized Learning Resource Recommendation System

📌 About the Project

LearnSmart AI is an AI-powered application that recommends personalized learning resources based on a learner's selected subject, skill level, and learning goal.

The project aims to help students find relevant learning materials more easily instead of spending time searching through many resources.

🎯 Problem Statement

Students often struggle to find suitable learning resources that match their current knowledge and learning objectives. Generic recommendations may not consider their skill level or specific goals.

LearnSmart AI addresses this problem by recommending learning resources according to the learner's preferences and learning needs.

💡 Proposed Solution

LearnSmart AI provides a desktop application where users can:

- Select a subject.
- Choose a learning level, such as Beginner.
- Enter a specific learning goal.
- Receive the top three recommended learning resources.
- View resource details and access the provided links.

✨ Key Features

- Personalized learning resource recommendations.
- Supports Python, Machine Learning, Web Development, Computer Science, and Data Analysis.
- Content-based recommendation using TF-IDF and cosine similarity.
- Java Swing desktop user interface.
- Python Flask backend API.
- Learning resources stored in a CSV dataset.

🛠️ Technologies Used

- Python: Main programming language.
- Streamlit: Web interface for the deployed application.
- Java Swing: Desktop user interface in the original version.
- Flask: Backend API in the original version.
- Pandas: Dataset handling and processing.
- Scikit-learn: TF-IDF vectorization and cosine similarity.
- CSV: Stores the learning resource dataset.
- GitHub: Source code management.
- Streamlit Community Cloud: Web application deployment and hosting.

⚙️ How It Works

1. The user selects a subject and learning level.
2. The user enters a learning goal.
3. The Java Swing application sends the request to the Flask API.
4. The Python backend compares the learning goal with available resources.
5. TF-IDF represents textual information as numerical vectors.
6. Cosine similarity measures how closely the resources match the user's learning needs.
7. The system displays the top three recommendations.

🧠 AI Algorithm

TF-IDF (Term Frequency–Inverse Document Frequency): Converts text describing learning resources into numerical vectors.

Cosine Similarity: Measures the similarity between the user's learning goal and the available learning resources.

The system uses these techniques for content-based recommendation.

🚀 How to Run the Project

Prerequisites

- Python installed
- Java Development Kit (JDK) installed
- IntelliJ IDEA or another Java IDE
- Visual Studio Code or another Python editor

Step 1: Install Python dependencies

Open a terminal in the "python-ai" folder and run:

pip install -r requirements.txt

Step 2: Start the Python backend

python app.py

Keep the Flask server running.

Step 3: Run the Java application

Open the Java project in IntelliJ IDEA and run "Main.java".

Step 4: Try the application

Select a subject, choose a learning level, enter a learning goal, and click the recommendation button.

🌍 Real-World Applications

- Personalized learning for college students.
- Self-paced online learning.
- Skill development and career preparation.
- Finding resources for specific technical topics.
- Supporting personalized educational platforms.

🔮 Future Scope

- User accounts and learning history.
- Feedback-based recommendation improvements.
- More subjects and learning resources.
- Integration with online learning platforms.
- Progress tracking and personalized learning paths.

🏆 Competition

Developed for the AI Project Competition organized by ACES, Department of Computer Engineering, Amrutvahini College of Engineering.

Theme: Education

Project: LearnSmart AI — Personalized Learning Resource Recommendation System.
