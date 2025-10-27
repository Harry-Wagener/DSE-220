# DSE 220: Project on Employee Burnout Turnover Prediction

Group members: 
- Grant Wagener (gwagener@ucsd.edu)
- Duy Nguyen (dnn007@ucsd.edu)
- Thomas Brehme (tbrehme@ucsd.edu)

# Description
Machine Learning Class Project with Thomas Brehme, Duy Nguyen -- This project will use the Employee Burnout & Turnover Prediction Dataset from HuggingFace to study the problem of predicting employee turnover.

---

## Table of Content:
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [References](#references)


## Abstract:
This project will use the Employee Burnout & Turnover Prediction Dataset (~850,000 records) from HuggingFace to study the problem of predicting employee turnover. The primary task is a supervised classification problem: given demographic, role, workload, sentiment, and performance features, we want to model the probability that an employee leaves the company. Our approach will begin with tabular baselines (e.g., logistic regression, tree-based models) and extend to multi-modal models that incorporate both structured and textual features. Model performance will be evaluated with standard classification metrics, and interpretability methods (e.g., feature importance) will be applied to identify key predictors of turnover. As a secondary option, we may also explore predicting burnout risk as a regression task, allowing us to compare its relationship to turnover. This dataset provides both scale and feature diversity, making it well-suited to our goal of building predictive and interpretable machine learning models.


## Dataset: 
**employee-burnout-turnover-prediction800**

    - Author: BrotherTony
    - Title: Synthetic Employee Dataset: 800K+ Records for HR Analytics
    - Year: 2025
    - Publisher: Hugging Face
    - URL:https://huggingface.co/datasets/BrotherTony/synthetic-employee-dataset


### Quick Information (from dataset website):
- Total Records	800,000+
- Departments	38 unique divisions
- Job Roles	300+ distinct positions
- Employee Personas	12 behavioral archetypes
- Features per Record	30+ attributes
- Salary Range	$27K - $384K


## Setup & Installation
Below are reproducible steps to set up a local environment and install the packages required to run the exploratory notebook.

Minimum requirements
- Python 3.12+
- ~2 GB free disk for a minimal environment; more RAM is recommended to load the full dataset (dataset ~850k rows, 1.3 GB)

Recommended setup (virtual environment)

```bash
# create virtual environment
python3 -m venv .venv
source .venv/bin/activate
# upgrade pip
python -m pip install --upgrade pip
```

Install the core Python packages used by the notebook:

```bash
pip install pandas numpy seaborn matplotlib jupyterlab notebook
```

Optional (for later modeling/experimentation):

```bash
pip install scikit-learn xgboost shap
```

You can also create a `requirements.txt` file containing the above packages and run `pip install -r requirements.txt` to reproduce the environment.

Notes:
- If you prefer Conda, create an environment with `conda create -n dse220 python=3.12` then `conda activate dse220` and use `pip` or `conda` to install packages.
- If you prefer to download the dataset, the code expects the relative path to the dataset is `data/employee-burnout-turnover-prediction-800k/synthetic-employee-dataset.json` (the notebook expects that relative path).
- If you prefer to not download the dataset and load it directly from Hugging Face, use this after using huggingface-cli:

```
# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_json("hf://datasets/BrotherTony/employee-burnout-turnover-prediction-800k/synthetic-employee-dataset.json")
```

## How to Run

There are two common workflows for running the analysis in `explore.ipynb`:

1) Run the notebook via Jupyter (Lab or classic Notebook)

From the repository root (after activating the virtualenv):

```bash
# start Jupyter Lab (recommended) or the classic notebook server
jupyter lab
# or
jupyter notebook
```

Open `explore.ipynb` in the browser UI and run the cells in order. The notebook includes an example data load:

```python
import pandas as pd
data_path = 'data/employee-burnout-turnover-prediction-800k/synthetic-employee-dataset.json'
df = pd.read_json(data_path)
```

2) Open the notebook inside Visual Studio Code

- Install the VS Code Python and Jupyter extensions.
- Open the repository folder in VS Code, open `explore.ipynb`, and select the Python kernel that corresponds to your virtual environment (look for the `.venv` interpreter).
- Run cells interactively from the notebook editor.

Troubleshooting / tips
- Loading the full JSON may be memory-intensive on low-RAM machines. If you run into memory errors, consider sampling the file or using chunked processing.
- If imports fail, confirm your currently active Python interpreter and that packages were installed into that interpreter (use `python -m pip list`).


## Exploratory Data Analysis

Here’s the correlation heatmap from the training dataset:

![Feature Correlation Heatmap](visualizations/heatmap.png)

## Modeling Approach
**Note:**
 _Our modeling implementation has not completed yet as that is a requirement of Milestone 3 for DSE220. As of 10/26/25, only the description of the approach is required._

Approach description:
We will split our dataset 80:20 test and train groups. We will maintain class proportions using the stratify argument in sklearn. 

Our target variable `left_company` has unequal distribution with 71% staying and 29% leaving for training. We will test a few methods to balance our target. Both SMOTE and ADASYN could perform well. We don't want to drop observations as some of our dependent variables have classes with very few members or distributions with long tails. 

The majority of our potential dependent variables are risk scores and ratings (ex: `performance_score`, `team_sentiment`) that are already standardized and will not require additional preprocessing for training. However, there are several below that we will address before training. 

| Variable | PreProcessing Steps |
| :--- | :--- |
| role | Transform - one hot encoding |
| job_level | Transform - one hot encoding |
| department | Transform - one hot encoding |
| tenure_months | Normalize |
| salary | Normalize |







## Results
Pending

## Future Work
Pending

## References
Pending

