🌳 Task-03 — Decision Tree Classifier for Customer Purchase Prediction

This repository contains Task-03 of my Data Science Internship at Prodigy InfoTech.
The goal of this task is to build a Decision Tree Classifier that predicts whether a customer will purchase a product or service based on their demographic and behavioral data.

🎯 Task Objective

The main objective is to apply machine learning classification techniques and understand how decision trees work for prediction tasks.

Using a dataset such as the Bank Marketing Dataset from the UCI Machine Learning Repository, the classifier predicts the customer’s likelihood of purchase (Yes/No).

📁 Sample Dataset

A sample dataset for this task is available here:
🔗 Dataset Link: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%203

(Datasets include customer age, job, marital status, balance, communication type, duration, campaign data, etc.)

🧪 Steps Performed in This Task
⿡ Data Loading & Understanding

Imported dataset using Pandas

Checked structure using .head(), .info(), .describe()

Identified input features (X) and target variable (y)

⿢ Data Cleaning

Handled missing values

Encoded categorical features (Label Encoding / One-Hot Encoding)

Normalized / scaled numerical values (if required)

⿣ Splitting Data

Divided dataset into:

Training Set (80%)

Testing Set (20%)

Used train_test_split() from sklearn

⿤ Model Building — Decision Tree Classifier

Used DecisionTreeClassifier() from sklearn

Trained the model on the training data

Predicted output on test data

⿥ Model Evaluation

Evaluated the model using:

Accuracy Score

Confusion Matrix

Classification Report

Visualization of Decision Tree (optional)

📊 Expected Output

The final output includes:

Prediction of whether a customer will purchase (Yes/No)

Performance metrics of the classifier

Visual interpretation of the decision tree

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook / Google Colab
