# Model Deployment – Gosling & Rottensteiner

This repository contains all three required phases of the assignment in English.

## Repository Structure

- `data/winequality-red.csv` – dataset used in all phases
- `phase1.ipynb` – Phase 1 notebook (scikit-learn training + web deployment export)
- `phase2.ipynb` – Phase 2 notebook (MLflow tracking + model registry + confidence interval logging)
- `phase2.pdf` – screenshots/discussion for MLflow comparison
- `phase3/streamlit_app.py` – local Streamlit dashboard using MLflow-registered model
- `phase3.pdf` – screenshots/discussion for Streamlit usage
- `docs/` – GitHub Pages files (HTML/CSS/JS)

## Phase 1 (scikit-learn + HTML/CSS/JS)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train/export browser model:

   ```bash
   python phase1_train.py
   ```

3. Open `docs/index.html` locally or deploy `docs/` with GitHub Pages.

GitHub Pages URL (project page):

- https://eocar.github.io/Model-Deployment-Gosling_Rottensteiner/
- Screenshot: https://github.com/user-attachments/assets/b84ac1ad-e14f-4e4a-afd7-d45a9e190956

## Phase 2 (MLflow local instance)

Run `phase2.ipynb` locally. It:

- creates an MLflow experiment
- logs parameters, metrics, artifacts, and tags
- logs confidence intervals (statsmodels OLS)
- registers the best model in the MLflow Model Registry

## Phase 3 (Streamlit local app)

1. Run `phase2.ipynb` first (to create and register model).
2. Start Streamlit:

   ```bash
   streamlit run phase3/streamlit_app.py
   ```

The app reads the registered MLflow model and displays prediction + confidence interval.
