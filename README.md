# 🚢 Titanic Survival Prediction

### End-to-End Machine Learning Project | Python • Scikit-learn • Pandas

## 📌 Project Overview

This project explores the famous Titanic dataset from Kaggle to predict whether a passenger survived the disaster based on demographic and travel-related information.

The goal was to build an end-to-end machine learning workflow, from exploratory data analysis and preprocessing to model training, evaluation, hyperparameter tuning, and generating predictions for Kaggle's test dataset.

The project uses **Logistic Regression** and Scikit-learn pipelines to create a reproducible classification workflow.

### 🏆 Project Results

| Metric | Result |
|---|---|
| Baseline Accuracy | 61.45% |
| Logistic Regression Validation Accuracy | 80.45% |
| Best GridSearchCV Accuracy | 80.20% |
| Tuned Model Validation Accuracy | 79.33% |
| **Kaggle Public Leaderboard Score** | **76.555%** |

The final Kaggle submission used Logistic Regression with `C=1.0`, trained on all 891 labeled passengers.

## 🎯 Project Objectives

- Explore the Titanic dataset and identify patterns related to survival.
- Handle missing values and categorical features.
- Build reusable preprocessing and machine learning pipelines.
- Train a Logistic Regression classification model.
- Evaluate performance using multiple classification metrics.
- Compare model performance against a baseline.
- Experiment with hyperparameter tuning using GridSearchCV.
- Generate predictions for Kaggle's unseen test dataset.

## 📊 Dataset

**Source:** [Kaggle Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

The dataset contains:

- **Training data:** 891 passengers, including survival outcomes.
- **Test data:** 418 passengers without survival outcomes.
- **Target variable:** `Survived` (0 = Did Not Survive, 1 = Survived).

The model uses seven features:

| Feature | Description |
|---|---|
| Pclass | Passenger ticket class |
| Sex | Passenger sex |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation |

## 🔎 Exploratory Data Analysis

The exploratory analysis revealed several interesting patterns.

### Survival by Sex

- Female survival rate: approximately **74.20%**.
- Male survival rate: approximately **18.89%**.

### Survival by Passenger Class

| Passenger Class | Survival Rate |
|---|---|
| First Class | 62.96% |
| Second Class | 47.28% |
| Third Class | 24.24% |

### Missing Values

The training dataset contained:

- `Age`: 177 missing values.
- `Cabin`: 687 missing values.
- `Embarked`: 2 missing values.

These findings helped guide feature selection and preprocessing decisions.

## ⚙️ Data Preprocessing

A Scikit-learn `ColumnTransformer` was used to process numerical and categorical features separately.

**Numerical features:**
- Missing values filled using median imputation.
- Features standardized using `StandardScaler`.

**Categorical features:**
- Missing values filled using most-frequent imputation.
- Categories converted to numerical representations using `OneHotEncoder`.

Preprocessing was integrated into the model pipeline to reduce data leakage and ensure consistent transformations.

## 🤖 Model Training

The primary classification algorithm was **Logistic Regression**.

The training dataset was divided into:

- 80% training data.
- 20% validation data.

The split used `random_state=42` and stratification to preserve the target class distribution.

### Model Evaluation

The original Logistic Regression model achieved **80.45% validation accuracy**.

Confusion matrix:

| | Predicted Not Survived | Predicted Survived |
|---|---:|---:|
| Actually Not Survived | 98 | 12 |
| Actually Survived | 23 | 46 |

The model performed better at identifying passengers who did not survive than at identifying survivors.

A `DummyClassifier` predicting the majority class achieved **61.45% accuracy**, providing a useful baseline.

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV` with five-fold cross-validation.

The following regularization values were evaluated:

`C = [0.01, 0.1, 1, 10, 100]`

**Results:**

- Best hyperparameter: `C=0.1`
- Best cross-validation accuracy: **80.20%**
- Tuned model validation accuracy: **79.33%**

Although the tuned model achieved the highest average cross-validation score among the tested configurations, it performed slightly worse than the original model on the held-out validation set.

This experiment demonstrated that hyperparameter tuning does not guarantee improved performance on every evaluation dataset.

## 🏆 Kaggle Submission

The final model was trained using all 891 labeled passengers and used to generate survival predictions for 418 test passengers.

The predictions were saved to `outputs/submission.csv` in the required Kaggle format.

**Kaggle Public Leaderboard Score: 0.76555 (76.555%)**

The difference between local validation accuracy and the public leaderboard score highlights the importance of evaluating models on independent data.

## 📁 Project Structure

```text
titanic-survival-prediction/
├── data/
│   ├── raw/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_final_predictions.ipynb
├── outputs/
│   └── submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

- **Python:** Programming language
- **Pandas and NumPy:** Data manipulation
- **Matplotlib and Seaborn:** Data visualization
- **Scikit-learn:** Preprocessing, model training, evaluation, and hyperparameter tuning
- **Jupyter Notebook:** Interactive development and experimentation
- **Git and GitHub:** Version control and project documentation

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/alinajdi225/titanic-survival-prediction.git
cd titanic-survival-prediction
```

Replace `YOUR_USERNAME` with the GitHub username of the repository owner.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/data).

Place the following files inside `data/raw/`:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

The raw datasets are excluded from version control.

### 5. Run the notebooks

Open the project in VS Code or Jupyter Notebook.

Execute the notebooks in numerical order, starting with exploratory analysis and ending with final predictions.

The final notebook generates `outputs/submission.csv`.

## 💡 Key Learnings

Through this project, I gained hands-on experience with:

- Building a complete supervised machine learning workflow.
- Investigating relationships between features and a classification target.
- Handling missing values and categorical variables.
- Preventing data leakage through Scikit-learn pipelines.
- Evaluating classification models beyond accuracy alone.
- Understanding cross-validation and regularization.
- Comparing local validation performance with Kaggle competition results.

## 🔮 Future Improvements

Potential next steps include:

- Experimenting with Random Forest and Gradient Boosting classifiers.
- Engineering additional features such as family size and passenger title.
- Comparing multiple models using consistent cross-validation.
- Improving model interpretability.
- Saving the final trained model for inference.

## 👤 About the Developer

I'm a developer with a background in web development who is expanding into AI and machine learning engineering.

This project represents a practical step in my learning journey, applying Python, data analysis, and machine learning concepts to a real classification problem.

I welcome feedback, collaboration, and opportunities to continue learning and building AI-powered solutions.