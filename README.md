# Spam Classification Comparison (Naïve Bayes vs Random Forest)

This Python script compares the performance of **Naïve Bayes** and **Random Forest** classifiers on the **Spambase dataset**. It evaluates key metrics such as accuracy, precision, and recall, and visualizes the results using **ROC curves** and a **bar graph**.

## 🚀 Features
- Loads and preprocesses the **Spambase dataset** using `ucimlrepo`
- Applies **VarianceThreshold** for feature selection (dimensionality reduction)
- Scales data using **StandardScaler**
- Splits the data into **training** and **test** sets (80/20 split)
- Trains and predicts using **Naïve Bayes** and **Random Forest** classifiers
- Evaluates models with **accuracy**, **precision**, and **recall**
- Visualizes model performance using **ROC curves** and a **bar graph** comparing accuracies

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** `ucimlrepo`, `scikit-learn`, `matplotlib`, `pandas`

## 🧪 Getting Started

### Prerequisites
Make sure you have **Python 3.x** installed and the required libraries. You can install them using:

```bash
pip install ucimlrepo scikit-learn matplotlib pandas

### Run the App
1. git clone https://github.com/Dylan7893/Python-Spam-Mail-Classification.git
