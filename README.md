# Project_F1_Analysis

# Project F1 Analysis & Race Winner Prediction

This repository contains a complete Formula 1 data analysis project, including exploratory data analysis (EDA), machine learning models for predicting race winners, and an interactive Streamlit application for visualizing insights and generating win-probability predictions.

# Overview

The project is built around real Formula 1 datasets (1950–2024) retrieved from Kaggle.
Two main components are included:

# Jupyter Notebook (ProjectF1.ipynb)

Data extraction and cleaning

Exploratory data analysis

Feature engineering

Machine learning model development

Model evaluation and interpretation

# Streamlit Application (app.py)

Interactive data exploration

Visual charts (top circuits, driver statistics, etc.)

Race-winner probability prediction using the trained XGBoost model

# Features
# Data Exploration

Most successful F1 drivers

Constructors with the most wins

Distribution of driver nationalities

Circuits hosting the most races

Seasonal changes in number of races

Relationship between qualifying position and race results

# Machine Learning

Target: Predict if a driver wins a race

Models built:

Logistic Regression

Random Forest

XGBoost (best model)

Evaluation metrics:

Accuracy

Classification report

Confusion matrix

Feature importance

# Streamlit App

Visual exploration dashboard

Top circuits chart

Driver comparison table

Input form for driver, grid position, year, and circuit

Real-time race-winner probability prediction



# Project Structure
Project_F1_Analysis/
│── ProjectF1.ipynb        # Data exploration & machine learning
│── app.py                 # Streamlit interactive application
│── README.md              # Project documentation
│── data/                  # Downloaded automatically via kagglehub

# Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-Learn

XGBoost

Streamlit

kagglehub


# Future Improvements

Add race-weather and pit-stop data

Add constructor-level predictive models

Build driver vs. driver comparison dashboards

