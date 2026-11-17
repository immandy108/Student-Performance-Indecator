# 🎓 Student Exam Performance Predictor

A complete, end-to-end Machine Learning project that predicts a student's **Math Score** based on factors like gender, race/ethnicity, parental education, lunch type, test preparation, reading score, and writing score. Built with Python, Flask, and Scikit-learn — with AWS Elastic Beanstalk deployment support.

---

## What This Project Does

You give the app some basic information about a student — their background, lunch type, whether they completed a test prep course, and their reading/writing scores — and it tells you what math score they're likely to get. That's it. Simple input, meaningful output.

Under the hood, it trains multiple regression models, picks the best one automatically using R² score, saves it as a `.pkl` file, and serves predictions through a Flask web form.

---

## Folder Structure 

```
mlproject/
│
├── application.py                # Flask entry point (AWS EB production)
├── setup.py                     # Package installer — registers `src` as installable
├── requirements.txt              # All Python dependencies
│
├── .ebextensions/
│   └── python.config             # AWS Elastic Beanstalk WSGI config
│
├── notebook/
│   ├── 1.EDA .ipynb              # Exploratory Data Analysis
│   ├── 2.Model_training.ipynb    # Manual model training experiments
│   └── data/
│       └── stud.csv              # Raw student dataset (source of truth)
│
├── src/                          # Core Python package
│   ├── __init__.py
│   ├── logger.py                 # Timestamped file logging setup
│   ├── exceptions.py             # Custom exception with file + line info
│   ├── utils.py                  # save_object, load_object, evaluate_model
│   │
│   ├── components/               # ML pipeline building blocks
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Reads CSV → splits train/test → saves to artifacts/
│   │   ├── data_transformation.py# Builds sklearn ColumnTransformer, saves preprocessor.pkl
│   │   └── model_trainer.py      # Trains 8 models, tunes with GridSearchCV, saves best
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py     # (stub — orchestration entry point)
│       └── predict_pipeline.py   # Loads model + preprocessor, returns prediction
│
├── artifacts/                    # Auto-generated during training (gitignored)
│   ├── raw_data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── templates/
│   ├── index.html                # Landing page
│   └── home.html                 # Prediction form + result display
│
└── logs/                         # Auto-generated per run (timestamped folders)
    └── MM_DD_YYYY_HH_MM_SS.log/
        └── MM_DD_YYYY_HH_MM_SS.log
```

---

## Models Compared

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — (baseline) |
| Decision Tree | criterion |
| Random Forest | n\_estimators |
| Gradient Boosting | learning\_rate, subsample, n\_estimators |
| AdaBoost | learning\_rate, n\_estimators |
| K-Nearest Neighbors | n\_neighbors, weights |
| XGBoost | learning\_rate, n\_estimators |
| CatBoost | depth, learning\_rate, iterations |

The best model is automatically selected by **test set R² score** and saved. If no model clears the **0.60 R²** threshold, the pipeline raises a `CustomException`.

---

## Setup & Run

### Clone and install

```bash
git clone <https://github.com/immandy108/Student-Performance-Indecator.git>
cd Student_Performance_Indecator
pip install -r requirements.txt
```

-> The `-e .` in `requirements.txt` runs `setup.py` automatically, so the `src` package is importable from anywhere in the project.

### Train the model

```bash
python src/components/data_ingestion.py
```

This runs the full pipeline in sequence: ingestion → transformation → training. After this, you'll see `artifacts/model.pkl` and `artifacts/preprocessor.pkl` created.

### Run the web app (local)

```bash
python application.py
```

Open your browser at `http://127.0.0.1:5000`. Go to `/predictdata` to fill in the form and get a prediction.

---

## API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page (Welcome screen) |
| `/predictdata` | GET | Renders the prediction form |
| `/predictdata` | POST | Accepts form data, returns predicted math score |

---


## Dependencies

```
pandas
numpy
seaborn
matplotlib
scikit-learn
catboost
xgboost
dill          # used for serializing sklearn pipelines (more robust than pickle)
flask
```

