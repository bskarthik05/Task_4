# Logistic Regression for Binary Classification

This Python script demonstrates a complete workflow for building and evaluating a logistic regression model for binary classification. It includes data loading, preprocessing, model training, evaluation, threshold tuning, and an explanation of the sigmoid function.

The script is designed to work with a dataset where the goal is to predict a binary outcome (e.g., malignant vs. benign tumors, spam vs. not spam).

## Table of Contents

* [Dataset](#-dataset)
* [Prerequisites](#-prerequisites)
* [Script Workflow](#-script-workflow)
* [Output](#-output)
* 
## Dataset

The script expects a CSV file named `data.csv` to be present in the same directory (or an accessible path).

**Important characteristics of the expected `data.csv`:**

1.  **Target Variable:** The script assumes a target variable column named `diagnosis`.
    * This column should initially contain categorical values like 'M' (for the positive class, e.g., Malignant) and 'B' (for the negative class, e.g., Benign).
    * The script maps these to `1` ('M') and `0` ('B'). If your target column has a different name or different values, you'll need to adjust the `TARGET_COL` variable and the mapping logic within the script.
2.  **Identifier Column:** The script explicitly drops a column named `id`. If your identifier column has a different name, adjust the script or ensure it's not present.
3.  **Empty Column:** The script explicitly drops a column named `Unnamed: 32`, which often appears in datasets saved from Excel and is typically empty.
4.  **Numeric Features:** Most other features are assumed to be numeric. The script includes a step to impute missing numerical values using the mean. Non-numeric features (other than the initial target and ID) would require additional preprocessing steps like one-hot encoding, which are mentioned in comments within the script but not fully implemented for simplicity in this example.

**If `data.csv` is not found, the script will print an error and exit.**

## ⚙️ Prerequisites

* Python 3.x
* The following Python libraries:
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `seaborn`
    * `scikit-learn`

You would typically install these libraries using a package manager like pip (e.g., by running `pip install pandas numpy matplotlib seaborn scikit-learn` in your terminal).

## Script Workflow

The script performs the following steps:

1.  **Load and Inspect Data:**
    * Loads the dataset from `data.csv`.
    * Prints the first 5 rows, dataset information (column types, non-null counts), descriptive statistics, and a count of missing values per column.
    * Drops the `Unnamed: 32` and `id` columns if they exist.

2.  **Data Preprocessing:**
    * **Missing Value Imputation:** For any numeric columns with missing values, it imputes them using the column's mean.
    * **Target Variable Handling:**
        * Identifies the `TARGET_COL` (assumed to be `diagnosis`).
        * Maps the target variable values: 'M' (Malignant) to `1` and 'B' (Benign) to `0`.
        * Performs checks to ensure the target variable is binary after mapping and contains no NaN values.
    * **Feature and Target Definition:**
        * Separates features (X) from the target variable (y).
        * Checks if features are numeric; if non-numeric features (other than the target) are present, it attempts to select only numeric ones.

3.  **Train/Test Split and Feature Standardization:**
    * Splits the data into training (80%) and testing (20%) sets. Stratification by the target variable `y` is used to ensure similar class proportions in both sets. A fixed `random_state` is used for reproducibility.
    * Standardizes the features using `StandardScaler` (fitting on the training set and transforming both training and test sets).

4.  **Fit a Logistic Regression Model:**
    * Initializes a `LogisticRegression` model (using `solver='liblinear'`, which is suitable for smaller datasets, and a fixed `random_state`).
    * Trains the model on the scaled training data.

5.  **Evaluate the Model:**
    * Makes predictions on the scaled test set.
    * Calculates prediction probabilities for the positive class.
    * **Confusion Matrix:** Displays and plots a confusion matrix to visualize true positives, true negatives, false positives, and false negatives.
    * **Classification Report:** Prints precision, recall, F1-score, and support for each class.
    * **ROC-AUC Score:** Calculates and prints the Area Under the Receiver Operating Characteristic Curve.
    * **ROC Curve:** Plots the ROC curve.

6.  **Tune Threshold and Explain Sigmoid Function:**
    * **Threshold Tuning:**
        * Explains that logistic regression's prediction method uses a default threshold of 0.5.
        * Demonstrates how varying this threshold (from 0.1 to 0.9) affects precision, recall, and F1-score for the positive class.
        * Plots the Precision-Recall curve against different thresholds, helping to visualize the trade-off.
    * **Sigmoid Function Explanation:**
        * Defines and plots the sigmoid (logistic) function: $S(z) = \frac{1}{1 + e^{-z}}$.
        * Explains how logistic regression uses a linear combination of inputs ($z$) and passes it through the sigmoid function to produce a probability.

## Output

The script will produce:

* **Console Output:**
    * Initial data exploration details (head, info, describe, null counts).
    * Messages about data cleaning and preprocessing steps (dropping columns, imputation).
    * Value counts for the target variable before and after mapping.
    * Shapes of training and test sets.
    * Confirmation of model training.
    * Confusion matrix (numeric).
    * Classification report.
    * ROC-AUC score.
    * Metrics (Precision, Recall, F1-score) for different classification thresholds.
    * Explanation of the sigmoid function's role.

* **Plots (displayed in separate windows or inline in environments like Jupyter):**
    * Confusion Matrix heatmap.
    * Receiver Operating Characteristic (ROC) Curve.
    * Precision and Recall vs. Threshold curve.
    * The Sigmoid (Logistic) Function graph.
