# loan_approval

https://www.kaggle.com/code/shayanzk/loan-approval-prediction


# Loan Approval Prediction Report

## Introduction
This report details the process of building a machine learning model to predict loan approval status based on various applicant features. The dataset contains information such as demographics, income, loan amount, and asset values. The objective is to analyze the data, preprocess it, explore relationships between features, and train several classification models to accurately predict whether a loan will be Approved or Rejected.

## Data Preprocessing

### Initial Data Inspection
The dataset initially contained 13 columns and 4269 entries. An initial check using `data.info()` confirmed that there were no missing values across any columns (`data.isnull().sum()` showed all zeros).

### Dropping 'loan_id'
The `loan_id` column was identified as a unique identifier with no predictive power and was therefore dropped from the dataset to avoid unnecessary complexity.

### Handling Categorical Features
Three columns were identified as categorical: `education`, `self_employed`, and `loan_status`. To prepare these for machine learning models, `LabelEncoder` from `sklearn.preprocessing` was used to convert these categorical text values into numerical representations. For example:
- `education`: 'Graduate' and 'Not Graduate' were converted to numerical labels.
- `self_employed`: 'Yes' and 'No' were converted to numerical labels.
- `loan_status`: 'Approved' and 'Rejected' were converted to numerical labels.

After this transformation, all columns in the dataset were of numerical type, suitable for model training.

## Exploratory Data Analysis (EDA)

### Categorical Variable Distributions
Before encoding, the distributions of the categorical variables (`education`, `self_employed`, `loan_status`) were visualized. This showed the count of each category within these features. For instance, there were more 'Graduate' applicants than 'Not Graduate' and more 'No' for 'self_employed' than 'Yes'. The target variable, `loan_status`, showed more 'Approved' loans than 'Rejected' ones.

### Correlation Heatmap
A correlation heatmap was generated to understand the relationships between all numerical features, including the newly encoded categorical ones. Key observations from the heatmap include:
- **`cibil_score`** showed a positive correlation with `loan_status`, indicating that a higher CIBIL score is generally associated with loan approval.
- **`income_annum`**, **`loan_amount`**, and various `assets_value` features (residential, commercial, luxury, bank) also showed varying degrees of correlation with `loan_status` and amongst themselves.
- Some asset values were strongly correlated with `income_annum` and `loan_amount`.

### Education, Self-Employed, and Loan Status Relationship
A categorical plot (`sns.catplot`) was used to visualize the relationship between `education`, `self_employed`, and `loan_status`. This plot helped to understand how combinations of education and self-employment status influence loan approval. For example, it might reveal that self-employed graduates have a different approval rate compared to non-self-employed graduates.

### Loan Amount vs. Income Annum
A scatter plot visualized the relationship between `income_annum` and `loan_amount`. This plot generally showed a positive trend, suggesting that individuals with higher annual incomes tend to apply for larger loan amounts. However, there was also significant spread, indicating other factors play a role in determining loan amounts.


## Model Training and Evaluation

### Train-Test Split
The dataset was split into training and testing sets to evaluate the models' performance on unseen data. A `test_size` of 0.4 (40%) was used, meaning 40% of the data was allocated for testing and 60% for training. `random_state=1` was set for reproducibility.

- **Training Set (X_train, Y_train):** 2561 samples
- **Testing Set (X_test, Y_test):** 1708 samples

### Models Used
Four different classification models were trained and evaluated:
1.  **RandomForestClassifier**
2.  **KNeighborsClassifier**
3.  **SVC (Support Vector Classifier)**
4.  **LogisticRegression**

### Model Performance (Accuracy Scores)
Each model was trained on the `X_train` and `Y_train` data and then evaluated on both the training and testing datasets using accuracy as the metric.

| Model                  | Training Accuracy (%) | Testing Accuracy (%) |
|:-----------------------|:----------------------|:---------------------|
| RandomForestClassifier | 97.93                 | 93.79                |
| KNeighborsClassifier   | 72.08                 | 58.72                |
| SVC                    | 61.77                 | 62.88                |
| LogisticRegression     | 72.04                 | 75.06                |

**Observations:**
- The **RandomForestClassifier** showed the highest accuracy on both the training set (97.93%) and the testing set (93.79%). This indicates excellent performance and generalization capability.
- **LogisticRegression** performed reasonably well on the testing set (75.06%), showing better generalization than K-Nearest Neighbors and SVC, despite a lower training accuracy.
- **KNeighborsClassifier** and **SVC** had lower accuracy scores compared to RandomForestClassifier and LogisticRegression, especially on the testing set for KNN.

Based on the testing accuracy, the **RandomForestClassifier** is the best-performing model for this dataset.

## Summary:

### Data Analysis Key Findings

*   **Data Preprocessing:** The initial dataset of 4269 entries and 13 columns was prepared by dropping the non-predictive 'loan\_id' column and converting three categorical features (`education`, `self_employed`, `loan_status`) into numerical representations using `LabelEncoder`.
*   **Exploratory Data Analysis (EDA) Insights:**
    *   `cibil_score` showed a positive correlation with `loan_status`, indicating that higher CIBIL scores are generally associated with loan approval.
    *   `income_annum` and `loan_amount` generally showed a positive relationship, suggesting higher income applicants tend to apply for larger loans.
    *   Distributions of categorical variables (`education`, `self_employed`, `loan_status`) were visualized, showing, for example, more 'Graduate' applicants and more 'Approved' loans than 'Rejected'.
*   **Model Performance:** Four classification models (RandomForestClassifier, KNeighborsClassifier, SVC, and LogisticRegression) were trained on a 60/40 train-test split (2561 training samples, 1708 testing samples).
    *   The **RandomForestClassifier** demonstrated the highest performance with a training accuracy of 97.93% and a testing accuracy of 93.79%.
    *   Logistic Regression achieved a testing accuracy of 75.06%, while KNeighborsClassifier and SVC had lower testing accuracies of 58.72% and 62.88%, respectively.

### Insights or Next Steps

*   The RandomForestClassifier is the most effective model for predicting loan approval in this dataset, given its high testing accuracy of 93.79%.
*   Future work should focus on hyperparameter tuning for the RandomForestClassifier, feature engineering to create more predictive variables, and exploring advanced models to potentially enhance accuracy further.
