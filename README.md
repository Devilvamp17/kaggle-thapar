# 🧠 ML Regression Contest – Ensemble Model with Optuna Tuning

This repository contains our solution for a Kaggle regression competition. We designed a powerful ensemble pipeline using stacking and fine-tuned it using Optuna. Our final submission leverages the top 13 most important features selected via permutation importance.

------------------------------------------------------------

📌 Key Highlights

- 📊 Task: Supervised Regression  
- 🧰 Tools: scikit-learn, XGBoost, LightGBM, Optuna  
- 🏗️ Models Used:
  - Linear Regression, Ridge, Bayesian Ridge
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost, LightGBM
  - Stacking Regressor as the final model

------------------------------------------------------------

🔁 Pipeline Overview

1. Data Preprocessing
   - Dropped unused columns (id, Row#)
   - Outlier removal using IQR method
   - Feature scaling using RobustScaler

2. Feature Selection
   - Trained a base stacking model
   - Computed permutation importance
   - Selected top 13 most important features

3. Model Ensembling with Optuna
   - Defined an Optuna objective to tune hyperparameters of all base models
   - Final StackingRegressor trained on tuned models

4. Inference & Submission
   - Processed test data
   - Used the final stack model to predict
   - Created submission.csv for Kaggle

------------------------------------------------------------

📂 File Structure

train.csv  
test.csv  
submission.csv
thapar.ipynb
thapar2.ipynb
final.py            # contains full training + prediction pipeline  
README.md
requirments.txt

------------------------------------------------------------

🚀 How to Run

pip install -r requirements.txt  
python main.py

This will:
- Train the Optuna-tuned ensemble
- Predict on test set
- Save results to submission.csv

------------------------------------------------------------

📈 Result

The model was evaluated using R² score with 5-fold cross-validation. Optuna helped achieve a more stable and accurate ensemble.

------------------------------------------------------------

📬 Contact

For questions or collaboration, feel free to reach out via issues or pull requests!
