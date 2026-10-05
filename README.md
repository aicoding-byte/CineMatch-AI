# 🎬 CineMatch AI — Movie Recommendation System

CineMatch AI is a movie recommendation web application built with **Python and Streamlit**. It recommends movies based on the movie selected by the user using content-based filtering and Natural Language Processing (NLP).

The application analyzes information such as movie overview, genres, keywords, director, and top cast members to identify movies with similar characteristics.

## ✨ Features

* 🎥 Search and select movies from the dataset
* 🤖 AI-based content recommendation
* 🔎 Uses NLP and TF-IDF for movie similarity
* 🎭 Considers genres, keywords, overview, director, and cast
* ⚡ Fast recommendation generation
* 🖥️ Interactive Streamlit web interface
* 🌙 Dark/light theme support
* 📊 Based on the TMDB 5000 Movies dataset

## 🧠 How It Works

CineMatch AI uses a **content-based recommendation approach**.

The system:

1. Loads movie information from the TMDB datasets.
2. Combines relevant movie information such as:

   * Overview
   * Genres
   * Keywords
   * Director
   * Top cast
3. Converts the combined text into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
4. Calculates similarity between movies using **cosine similarity**.
5. Returns movies that are most similar to the movie selected by the user.

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TF-IDF Vectorization**
* **Cosine Similarity**
* **TMDB 5000 Movie Dataset**

## 📂 Project Structure

```text
CineMatch-AI-Movie-Recommender/
│
├── app.py
├── requirements.txt
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── .gitignore
└── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/aicoding-byte/CineMatch-AI-Movie-Recommender.git
```

### 2. Open the project folder

```bash
cd CineMatch-AI-Movie-Recommender
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset

This project uses the **TMDB 5000 Movies Dataset**, which contains movie information used to generate content-based recommendations.

The dataset includes information such as movie titles, overviews, genres, keywords, cast, and crew.

## 🎯 Project Objective

The objective of CineMatch AI is to demonstrate how **Artificial Intelligence, Natural Language Processing, and machine learning techniques** can be used to build a practical movie recommendation system.

## 🔮 Future Improvements

Possible future improvements include:

* Integration with the TMDB API for live movie posters and information
* Personalized recommendations based on user preferences
* User accounts and recommendation history
* Improved recommendation algorithms
* More detailed movie information
* Deployment as a public web application

## 👨‍💻 Author

Developed as an **M.Sc. Artificial Intelligence academic project**.

---

⭐ If you found this project interesting, consider giving the repository a star!
