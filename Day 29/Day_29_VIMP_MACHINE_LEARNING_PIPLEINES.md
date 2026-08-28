# 🚀 Day 29: Machine Learning Pipelines — The Complete Guide

> A comprehensive, step-by-step masterclass on building, evaluating, exporting, and deploying Machine Learning workflows with and without Scikit-Learn Pipelines.

---

## 📑 Table of Contents
1. [Core Concept: What is an ML Pipeline?](#-core-concept-what-is-an-ml-pipeline)
2. [High-Level Architecture Comparison](#-high-level-architecture-comparison)
3. [Notebook 1: Machine Learning Without Pipelines (`1_machine_learning_without_piplines.ipynb`)](#-notebook-1-machine-learning-without-pipelines)
4. [Notebook 2: Inference Without Pipelines (`1_prediction_exported_model.ipynb`)](#-notebook-2-inference-without-pipelines)
5. [Notebook 3: Machine Learning With Pipelines (`2_machine_learning_with_piplining.ipynb`)](#-notebook-3-machine-learning-with-pipelines)
6. [Notebook 4: Inference With Pipelines (`2_predicted_output_ml_pipeline.ipynb`)](#-notebook-4-inference-with-pipelines)
7. [Side-by-Side Feature Matrix](#-side-by-side-feature-matrix)
8. [Best Practices & Golden Rules](#-best-practices--golden-rules)

---

## 🧠 Core Concept: What is an ML Pipeline?

In real-world machine learning, building a model is rarely just about calling `model.fit(X, y)`. Before feeding data into any algorithm, you almost always need to:
1. **Impute missing values** (mean, median, mode).
2. **Encode categorical variables** (One-Hot Encoding, Ordinal Encoding).
3. **Scale numerical features** (StandardScaler, MinMaxScaler).
4. **Select key features** (SelectKBest, PCA).
5. **Train the estimator** (Decision Tree, Random Forest, Logistic Regression).

```
   Raw Data
      │
      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Imputation  │ ───► │  Encoding    │ ───► │   Scaling    │ ───► │  Estimator   │ ───► Predictions
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

### The Problem with the Traditional (Manual) Approach:
* **Data Leakage:** Preprocessing the whole dataset before train-test split leaks test information into training.
* **Messy Code:** You have to track dozens of intermediate NumPy arrays and DataFrames (`X_train_age`, `X_train_sex`, `X_train_embarked`, etc.).
* **Deployment Hell:** In production, every single transformer must be manually loaded, kept in strict sync, and executed in the exact correct order. If one column order shifts by 1 index, predictions silently fail or turn into garbage.

### The Solution: `sklearn.pipeline.Pipeline`
A **Pipeline** chains multiple transformers and an estimator into a single, unified Scikit-Learn object. Calling `pipe.fit()` automatically executes `fit_transform()` through all preprocessing steps and `fit()` on the model. Calling `pipe.predict()` automatically transforms raw data and outputs predictions.

---

## 🔄 High-Level Architecture Comparison

```
Traditional Approach (Without Pipeline)
---------------------------------------
Raw Input ──► [Manual Imputer] ──► [Manual OHE] ──► [Manual Concatenate] ──► [clf.predict()]
(Must export and manage 3-5 separate .pkl files and write custom preprocessing logic in production)

Pipeline Approach (With Pipeline)
---------------------------------
Raw Input ───────────────────────► [ pipe.predict(raw_input) ] ──────────────────────► Prediction
(Exports only 1 .pkl file; all preprocessing + model logic encapsulated internally)
```

---

## 🧪 Notebook 1: Machine Learning Without Pipelines
**File:** `1_machine_learning_without_piplines.ipynb`  
**Goal:** Train a Titanic survival classifier using traditional manual preprocessing and export each component separately.

### Step 1: Data Loading & Initial Cleanup
We use the classic Titanic dataset (`train.csv`):
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('train.csv')

# Drop columns that have high cardinality or no predictive value
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
```

### Step 2: Train-Test Split (Preventing Data Leakage)
We split data **before** calculating any imputation or scaling statistics:
```python
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 3: Manual Feature Transformation
Each feature group requires individual transformers:

```python
# 1. Imputing Missing Values in 'Age' (using mean)
si_age = SimpleImputer()
X_train_age = si_age.fit_transform(X_train[['Age']])
X_test_age = si_age.transform(X_test[['Age']])

# 2. Imputing Missing Values in 'Embarked' (using most frequent)
si_embarked = SimpleImputer(strategy='most_frequent')
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])
X_test_embarked = si_embarked.transform(X_test[['Embarked']])

# 3. One-Hot Encoding 'Sex'
ohe_sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_test_sex = ohe_sex.transform(X_test[['Sex']])

# 4. One-Hot Encoding 'Embarked'
ohe_embarked = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)
X_test_embarked = ohe_embarked.transform(X_test_embarked)

# 5. Extracting Remaining Numerical Columns
X_train_rem = X_train[['Pclass', 'SibSp', 'Parch', 'Fare']].values
X_test_rem = X_test[['Pclass', 'SibSp', 'Parch', 'Fare']].values
```

### Step 4: Concatenating Transformed Arrays
We manually stack all individual feature matrices horizontally:
```python
X_train_transformed = np.concatenate(
    (X_train_rem, X_train_age, X_train_sex, X_train_embarked), axis=1
)

X_test_transformed = np.concatenate(
    (X_test_rem, X_test_age, X_test_sex, X_test_embarked), axis=1
)
```

### Step 5: Training and Evaluating Model
```python
clf = DecisionTreeClassifier()
clf.fit(X_train_transformed, y_train)

y_pred = clf.predict(X_test_transformed)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Step 6: Exporting Artifacts with Pickle
Because preprocessing and training were decoupled, we must export multiple individual files:
```python
pickle.dump(ohe_sex, open('ohe_sex.pkl', 'wb'))
pickle.dump(ohe_embarked, open('ohe_embarked.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
```

> [!WARNING]
> **Key Drawbacks of Notebook 1:**
> - High chance of index mismatch or column transposition during concatenation.
> - We forgot to save `si_age` and `si_embarked`, which will cause issues when unseen test records contain missing values!
> - Maintaining 3-5 pickle files creates operational overhead in production.

---

## 📦 Notebook 2: Inference Without Pipelines
**File:** `1_prediction_exported_model.ipynb`  
**Goal:** Simulate a production server receiving raw user data and making a prediction using the exported models from Notebook 1.

### Step 1: Loading All Exported Artifacts
```python
import pickle
import numpy as np

# Load individual transformers and model
ohe_sex = pickle.load(open('ohe_sex.pkl', 'rb'))
ohe_embarked = pickle.load(open('ohe_embarked.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))
```

### Step 2: Receiving Raw User Input
Suppose a user submits data from a web form:
```python
# User Profile: Pclass=2, Sex='male', Age=31.0, SibSp=0, Parch=0, Fare=10.5, Embarked='S'
user_input = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)
```

### Step 3: Manually Replicating Training Transformations
The production engineer must recreate the exact same feature engineering pipeline manually in Python:
```python
# Extract and transform 'Sex' (Index 1)
sex_transformed = ohe_sex.transform(user_input[:, 1].reshape(1, 1))

# Extract and transform 'Embarked' (Index 6)
embarked_transformed = ohe_embarked.transform(user_input[:, 6].reshape(1, 1))

# Extract numerical columns: Pclass(0), SibSp(3), Parch(4), Fare(5)
rem_features = user_input[:, [0, 3, 4, 5]]

# Extract Age (Index 2)
age_feature = user_input[:, 2].reshape(1, 1)

# Concatenate in the EXACT SAME order as Notebook 1: [rem, age, sex, embarked]
final_input = np.concatenate((rem_features, age_feature, sex_transformed, embarked_transformed), axis=1)
```

### Step 4: Making Prediction
```python
prediction = clf.predict(final_input)

if prediction[0] == 1:
    print("Passenger would have Survived.")
else:
    print("Passenger would NOT have Survived.")
```

> [!CAUTION]
> **Why this manual inference is dangerous in production:**
> 1. **Brittle Code:** If a data scientist reorders columns in training, the production script breaks silently.
> 2. **Code Duplication:** The entire transformation logic must be written twice (once in training, once in inference).
> 3. **Missing Imputer Disaster:** If the user leaves `Age` blank (`NaN`), this script crashes because we didn't export `si_age`.

---

## ⚡ Notebook 3: Machine Learning With Pipelines
**File:** `2_machine_learning_with_piplining.ipynb`  
**Goal:** Build a robust, automated end-to-end Machine Learning Pipeline using Scikit-Learn's `ColumnTransformer` and `Pipeline`.

### Step 1: Loading Data & Train-Test Split
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
import pickle

df = pd.read_csv('train.csv')
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Survived']), 
    df['Survived'], 
    test_size=0.2, 
    random_state=42
)
```

### Step 2: Defining Modular Steps with `ColumnTransformer`
We create clear, declarative transformation blocks:

```python
# -------------------------------------------------------------
# Transformer 1: Missing Value Imputation
# -------------------------------------------------------------
# Age is at index [2], Embarked is at index [6]
trf1 = ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])
], remainder='passthrough')

# -------------------------------------------------------------
# Transformer 2: Categorical Encoding (One-Hot Encoding)
# -------------------------------------------------------------
# Sex is at index [1], Embarked is at index [6]
trf2 = ColumnTransformer([
    ('ohe_sex', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1]),
    ('ohe_embarked', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [6])
], remainder='passthrough')

# -------------------------------------------------------------
# Transformer 3: Feature Scaling
# -------------------------------------------------------------
# Scale all resulting numeric columns (0 to 10)
trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0, 10))
])

# -------------------------------------------------------------
# Transformer 4: Feature Selection
# -------------------------------------------------------------
# Select top 8 best features based on Chi-Squared test
trf4 = SelectKBest(score_func=chi2, k=8)

# -------------------------------------------------------------
# Step 5: Classifier Algorithm
# -------------------------------------------------------------
trf5 = DecisionTreeClassifier()
```

### Step 3: Constructing the Master Pipeline
We assemble all stages into a single sequential execution pipeline:

```python
pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', trf4),
    ('trf5', trf5)
])
```

#### Visualizing Pipeline in Jupyter:
```python
from sklearn import set_config
set_config(display='diagram')
pipe  # Displays interactive HTML diagram of all components
```

### Step 4: Training & Evaluating in One Command
```python
# Fitting the entire pipeline
pipe.fit(X_train, y_train)

# Predicting on test data
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Step 5: Cross-Validation Without Data Leakage
When cross-validating a pipeline, scikit-learn fits the transformers **only on the training folds** of each split, guaranteeing zero data leakage:
```python
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
print("5-Fold CV Mean Accuracy:", cv_scores.mean())
```

### Step 6: Hyperparameter Tuning with `GridSearchCV`
To tune parameters of individual steps inside a pipeline, use the `<step_name>__<parameter>` syntax (with **two underscores `__`**):

```python
params = {
    'trf5__max_depth': [1, 2, 3, 4, 5, None],
    'trf5__criterion': ['gini', 'entropy']
}

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
```

### Step 7: Exporting the Entire Pipeline
Instead of exporting 4 separate files, we export a **single file** containing all transformers, parameters, feature selectors, and the model:
```python
pickle.dump(pipe, open('pipe.pkl', 'wb'))
```

---

## 🎯 Notebook 4: Inference With Pipelines
**File:** `2_predicted_output_ml_pipeline.ipynb`  
**Goal:** Load the trained pipeline artifact and make predictions on new data with zero manual preprocessing.

### Step 1: Load the Single Pipeline Artifact
```python
import pickle
import numpy as np

# Load the single unified pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))
```

### Step 2: Pass Raw Data Directly
```python
# Raw test passenger: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
test_input = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)

# Predict directly with 1 line of code!
prediction = pipe.predict(test_input)

print("Prediction (0 = Not Survived, 1 = Survived):", prediction[0])
```

> [!TIP]
> **Why Notebook 4 is a Game Changer:**
> * **Zero Preprocessing Code:** No manual encoding, no manual imputation, no manual scaling, no array reshaping/slicing.
> * **Guaranteed Consistency:** Exactly identical transformations are applied as during training.
> * **Production Ready:** Only a single `.pkl` file to deploy to Flask/FastAPI/AWS Lambda.

---

## 📊 Side-by-Side Feature Matrix

| Feature / Criteria | ❌ Without Pipelines (Notebooks 1 & 2) | ✅ With Pipelines (Notebooks 3 & 4) |
| :--- | :--- | :--- |
| **Exported Files** | 3 to 5 `.pkl` files (`ohe_sex`, `ohe_embarked`, `clf`...) | Exactly 1 `.pkl` file (`pipe.pkl`) |
| **Inference Code Lines** | ~20+ lines of manual array manipulation | **1 line:** `pipe.predict(raw_data)` |
| **Data Leakage Risk** | High (easy to fit transformers on full data) | **Zero** (automatically handled in CV) |
| **Cross-Validation** | Complex and tedious to write manually | Simple: `cross_val_score(pipe, ...)` |
| **Hyperparameter Tuning**| Hard to tune preprocessing & model together | Easy via `GridSearchCV(pipe, params)` |
| **Debugging & Maintenance**| High risk of column shifting bugs | Modular, encapsulated, and easy to audit |
| **Readability** | Cluttered with intermediate variables | Clean, elegant, and self-documenting |

---

## 🏆 Best Practices & Golden Rules

1. **Use Column Indices for Intermediate Pipeline Steps:**
   When a step transforms data into a NumPy array, column names are lost. For downstream `ColumnTransformer` steps, reference columns by integer index (e.g. `[2]`, `[6]`) or `slice()`.

2. **Always Use Double Underscores in `GridSearchCV`:**
   When setting parameters for a pipeline step, the syntax is strictly `step_name__parameter_name` (e.g. `trf5__max_depth`, not `trf5_max_depth`).

3. **Use `set_config(display='diagram')`:**
   Enabling diagram display in Jupyter gives you a visual schematic of your entire ML flow.

4. **Combine with `make_pipeline` for Simple Flows:**
   If you don't need custom step names, `make_pipeline(trf1, trf2, model)` automatically generates step names for cleaner code.
