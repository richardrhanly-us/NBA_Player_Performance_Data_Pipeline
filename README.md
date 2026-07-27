# NBA Player Performance Prediction

[![Streamlit Health Check](https://github.com/richardrhanly-us/NBA_Player_Performance_Data_Pipeline/actions/workflows/streamlit-health-check.yml/badge.svg)](https://github.com/richardrhanly-us/NBA_Player_Performance_Data_Pipeline/actions/workflows/streamlit-health-check.yml)
[![Python Code Check](https://github.com/richardrhanly-us/NBA_Player_Performance_Data_Pipeline/actions/workflows/python-check.yml/badge.svg)](https://github.com/richardrhanly-us/NBA_Player_Performance_Data_Pipeline/actions/workflows/python-check.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://edgeanalyticsnba.streamlit.app/)

A Python data pipeline and machine-learning application for analyzing NBA player performance and evaluating points-based betting props.

<p align="center">
  <a href="https://edgeanalyticsnba.streamlit.app/">
    <img
      src="images/App2Capture.PNG"
      alt="NBA Player Performance Prediction application"
      width="1000"
    />
  </a>
</p>

---

## Overview

This project combines NBA game data, feature engineering, regression modeling, and an interactive Streamlit application into a complete player-performance prediction workflow.

The application allows users to:

- Search for an NBA player
- Review recent player-performance trends
- View the player’s current matchup when available
- Enter a proposed points line
- Generate a model-based scoring prediction
- Estimate the probability of finishing over or under the line
- Receive a simplified lean based on the predicted edge

The project was built to demonstrate an end-to-end data application rather than an isolated machine-learning notebook.

---

## Live Application

The public application is available here:

### [Launch NBA Player Performance Prediction](https://edgeanalyticsnba.streamlit.app/)

The deployment is monitored by a scheduled GitHub Actions health check.

Because NBA games are seasonal, live matchup information may be limited during the offseason. Historical player analysis and model functionality may still be available depending on the current data source.

---

## Key Features

### Player Search

Users can search for an NBA player and retrieve recent game-log information through the application interface.

### Performance Analysis

The application calculates recent and long-term performance indicators, including scoring, shot volume, free-throw opportunities, minutes played, and overall game productivity.

### Model-Based Prediction

A trained regression model uses engineered player features to estimate expected points for the selected player.

### Prop-Line Evaluation

Users can enter a points line and compare it with the model’s prediction.

The application converts the estimated difference into a probability-oriented result and displays a simplified recommendation:

- Lean Over
- Lean Under
- No Significant Edge

### Interactive Streamlit Interface

The model and supporting data are presented through a publicly accessible web application designed for users who do not need to interact directly with Python code.

---

## Data Pipeline

The project follows this general workflow:

```text
NBA game logs
      ↓
Data collection
      ↓
Cleaning and validation
      ↓
Feature engineering
      ↓
Model training and evaluation
      ↓
Saved prediction model
      ↓
Streamlit application
      ↓
Player prediction and prop evaluation
