## 🧠 System Architecture & Functionality

COLAB LINK :- https://colab.research.google.com/drive/1NbTqpY_7UrasrFZdttIXCiD8aSHYbnJG#scrollTo=c09dc557

### In Short
The **AutoML Benchmarking Bot** is an end-to-end Python pipeline that transforms raw CSV data into a documented, optimized model. It automates the 'boring stuff'—imputation, encoding, hyperparameter search, and reporting—using industry-standard libraries.

### Tech Stack
*   **Optimization Engine:** [Optuna](https://optuna.org/) (Bayesian Optimization via TPESampler).
*   **ML Frameworks:** Scikit-Learn (RandomForest, Linear/Logistic Baselines) and XGBoost.
*   **Data Processing:** Pandas and NumPy.
*   **Visualization:** Matplotlib and Seaborn.
*   **Environment:** Google Colab / IPython for interactive widgets and Markdown rendering.

### Detailed Functionality

#### 1. Smart Data Ingestion
- Uses `google.colab.files` for local file uploads.
- Detects the nature of the target variable: if the target is an object (string) or has low cardinality (<20 unique values), it automatically switches to a **Classification** task; otherwise, it handles it as **Regression**.

#### 2. Automated Preprocessing Pipeline
- **Imputation:** Automatically fills missing numerical values using the `median` to remain robust against outliers.
- **Categorical Encoding:** Converts text labels into numerical features using `pd.get_dummies` (one-hot encoding) to make them compatible with XGBoost and Random Forest.
- **Validation Strategy:** Splits data into 80/20 Train/Test sets to ensure unbiased evaluation.

#### 3. The Optuna Tuning Engine
Instead of a slow 'Grid Search', the bot uses **Bayesian Optimization**:
- **Random Forest:** Tunes `n_estimators` and `max_depth`.
- **XGBoost:** Tunes `learning_rate`, `max_depth`, and `n_estimators`.
- **Cross-Validation:** Each trial uses 3-fold CV on the training set to prevent overfitting during the search.

#### 4. Automated Documentation (Model Card)
- **Baseline Comparison:** Compares the complex models against a 'simple' baseline (Logistic/Linear Regression). If the complex model isn't significantly better, you know immediately.
- **Interpretability:** Generates a horizontal bar chart of **Feature Importance**, showing which columns actually drive the predictions.
- **Model Card:** Uses the `IPython.display.Markdown` module to generate a professional report that can be copy-pasted directly into project documentation.

### Why it's 18x Faster (3 hrs -> 10 mins)
1.  **Parallel Search:** Optuna intelligently 'prunes' bad trials and focuses on promising hyperparameter regions.
2.  **No Manual Refactoring:** You don't have to rewrite the pipeline when switching from a classification dataset to a regression dataset.
3.  **Instant Reporting:** The time spent manually calculating metrics and formatting tables is reduced to milliseconds.
