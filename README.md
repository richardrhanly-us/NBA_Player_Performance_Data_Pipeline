# NBA Player Performance Prediction

This project applies machine learning techniques to predict NBA player performance using game-level statistics from the 2024–2025 NBA season.

The goal is to explore how historical player performance can be used to forecast future outcomes while avoiding data leakage and maintaining a realistic prediction setting.

The project analyzes player statistics, engineers features using rolling game data, and trains models to predict overall performance and scoring outcomes.

---

## Project Objectives

The notebook focuses on two prediction tasks.

Regression  
Predict a player's Game Score (GmSc), an advanced metric that summarizes overall player performance in a single value.

Classification  
Predict whether a player will score 20+ points or 25+ points in a game.

Rolling statistics from previous games are used so the model only relies on information that would have been available before the game occurred.

---

## Dataset

Dataset: NBA Player Stats 2024–2025  
Source: Kaggle

The dataset contains game-level box score statistics including:

- points
- rebounds
- assists
- minutes played
- shooting statistics
- team and player information

Each row represents a player’s performance in a single game.

---

## Project Workflow

The project follows a typical machine learning pipeline.

1. Load dataset
2. Inspect and clean data
3. Feature engineering using rolling statistics
4. Exploratory data analysis
5. Train regression and classification models
6. Evaluate model performance
7. Interpret results

---

## Repository Structure

nba-player-performance/
│
├── data/
│   └── database_24_25.csv
│
├── notebook/
│   └── nba_analysis.ipynb
│
├── README.md
└── requirements.txt

---

## Tools and Libraries

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib

---

## Running the Project

Clone the repository

git clone https://github.com/yourusername/nba-player-performance.git

Install dependencies

pip install -r requirements.txt

Run the notebook

jupyter notebook

Open the notebook and run the cells sequentially.

---

## Future Improvements

Possible extensions for this project include:

- incorporating team defensive metrics
- predicting additional performance statistics
- using advanced models such as gradient boosting
- building a player performance dashboard

---

## Author

Ray Hanly  
Computer Information Technology  
Machine Learning / Data Analysis Project
