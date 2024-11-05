# 🏡 Real Estate Price Prediction - Immo Eliza

## 📑 Table of Contents

1. [🔎 Project Overview](#project-overview)
2. [⏱️ Project Timeline](#project-timeline)
3. [📊 Dataset](#dataset)
4. [🤖 Model Training](#model-training)
5. [📈 Performance](#performance)
6. [👥 Contributor](#contributor)

## 🔎 Project Overview

This project aims to predict real estate prices in Belgium using various machine learning models. The primary objective is to provide accurate price estimates for properties based on their features like location, area, number of bedrooms, etc.

## ⏱️ Project Timeline
The initial setup of this project was completed in 5 days.

## 📊 Dataset

The dataset used in this project contains information about real estate properties in Belgium, including details such as property type, location, living area, number of bedrooms, and more. It comprises around 76,000 houses.

## 🤖 Model training

The project explores several machine learning models, starting with a baseline RandomForest model and experimenting with other models like Linear Regression and XGBoost Regressor. The final model selection is based on performance metric R² score.

It is found that the RandomForest Regressor and XGBoost Regressor models performed the best.

Here is an example of the evaluation results for the test set, with four different algorithms:

```

| Model           | R2    |
|                 |       |
| LinearRegression| 0.36  |
| XGBoost         | 0.72  |
| Random Forest   | 0.99  |

## 📈 Performance

The best-performing model achieved an R² score of 0.72 on the test set, indicating that it can explain 72% of the variance in property prices.

## 👥 Contributor

Minh Duc Nguyen - https://www.linkedin.com/in/minhducnguyen63/
