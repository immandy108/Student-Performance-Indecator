# 🎓 Student Exam Performance Predictor

A complete, end-to-end Machine Learning project that predicts a student's **Math Score** based on factors like gender, race/ethnicity, parental education, lunch type, test preparation, reading score, and writing score. Built with Python, Flask, and Scikit-learn — with AWS Elastic Beanstalk deployment support.

---

## 📌 What This Project Does

You give the app some basic information about a student — their background, lunch type, whether they completed a test prep course, and their reading/writing scores — and it tells you what math score they're likely to get. That's it. Simple input, meaningful output.

Under the hood, it trains multiple regression models, picks the best one automatically using R² score, saves it as a `.pkl` file, and serves predictions through a Flask web form.

---

## 🗂️ Folder Structure

```
mlproject/
│
├── app.py                        # Flask entry point (local dev with debug=True)
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

## ⚙️ How the ML Pipeline Works

```
notebook/data/stud.csv
        │
        ▼
  DataIngestion              → splits into train.csv / test.csv
        │
        ▼
  DataTransformation         → numerical: median impute + StandardScaler
                               categorical: mode impute + OneHotEncoder + StandardScaler
                               saves preprocessor.pkl
        │
        ▼
  ModelTrainer               → trains 8 models with GridSearchCV (3-fold CV)
                               picks best R² score (threshold: > 0.60)
                               saves model.pkl
        │
        ▼
  PredictPipeline            → loads model.pkl + preprocessor.pkl
                               transforms input → returns math score
```

---

## 🤖 Models Compared

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

## 🚀 Setup & Run

### 1. Clone and install

```bash
git clone <your-repo-url>
cd mlproject
pip install -r requirements.txt
```

> The `-e .` in `requirements.txt` runs `setup.py` automatically, so the `src` package is importable from anywhere in the project.

### 2. Train the model

```bash
python src/components/data_ingestion.py
```

This runs the full pipeline in sequence: ingestion → transformation → training. After this, you'll see `artifacts/model.pkl` and `artifacts/preprocessor.pkl` created.

### 3. Run the web app (local)

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`. Go to `/predictdata` to fill in the form and get a prediction.

---

## 🌐 API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page (Welcome screen) |
| `/predictdata` | GET | Renders the prediction form |
| `/predictdata` | POST | Accepts form data, returns predicted math score |

---

## 🏗️ Deployment — AWS Elastic Beanstalk

This project is already set up for AWS EB deployment.

The `.ebextensions/python.config` file tells Elastic Beanstalk to use `application.py` as the WSGI entry point:

```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

**Key distinction:**
- `app.py` — local dev, runs with `debug=True`
- `application.py` — production, runs without debug mode (required by AWS EB)

To deploy:
```bash
eb init
eb create
eb deploy
```

---

## 📦 Dependencies

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

---

## 📊 Project Health Scorecard

| Area | Score | Notes |
|---|---|---|
| **Folder Structure** | ✅ 9/10 | Clean separation of `src/`, `notebook/`, `templates/`, `artifacts/` |
| **Path Accuracy** | ⚠️ 7/10 | Hardcoded relative paths (`artifacts/model.pkl`, `notebook/data/stud.csv`) — works fine if you run from project root, breaks if you don't |
| **ML Pipeline Design** | ✅ 9/10 | Proper ingestion → transformation → training chain, preprocessor saved separately |
| **Model Selection** | ✅ 8/10 | 8 models compared, GridSearchCV tuning, threshold guard at R² > 0.6 |
| **Error Handling** | ✅ 8/10 | Custom exception class captures filename + line number |
| **Logging** | ✅ 8/10 | Per-run timestamped log files, though log folders accumulate (57+ already) |
| **Flask App** | ✅ 8/10 | Clean routes, two entry points for local vs. production |
| **Deployment Readiness** | ✅ 9/10 | AWS EB config present, WSGI path set correctly |
| **HTML Templates** | ⚠️ 6/10 | Labels and input `name` attributes are swapped (reading/writing field labels mixed up) |
| **Code Quality** | ✅ 7/10 | Well-structured, though `train_pipeline.py` is currently empty |

> **Overall: ~79/100** — A solid, well-structured ML project. Main things to revisit: absolute path handling, the label swap in `home.html`, and filling out `train_pipeline.py`.

---

## ⚠️ Known Issues

1. **Label mismatch in `home.html`** — The "Writing Score" label points to `reading_score` input field and vice versa. The names are swapped. The backend logic is correct; it's just a display bug.

2. **Hardcoded relative paths** — `predict_pipeline.py` uses `'artifacts/model.pkl'` as a plain string. If the app is ever run from a different working directory, it will fail with a `FileNotFoundError`. Use `os.path.join(os.path.dirname(__file__), ...)` for robustness.

3. **`train_pipeline.py` is empty** — The file exists but has no code. The training is triggered directly by running `data_ingestion.py`. A proper orchestration script here would make things cleaner.

4. **Log folder accumulation** — Each run creates a new folder inside `logs/`. After heavy development, you end up with 57+ folders. Consider cleaning these up periodically.

---

## 👤 Author

**Mandeep Kumar**
📧 immandy108@gmail.com

---

## 📄 License

This project is for educational and portfolio purposes.
