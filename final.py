import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.inspection import permutation_importance
import optuna

# 1. Load and preprocess training data
df = pd.read_csv('train.csv')
df.drop(columns=['id', 'Row#'], inplace=True, errors='ignore')

# Remove outliers
q1, q3 = df["output"].quantile([0.25, 0.75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df = df[(df["output"] >= lower) & (df["output"] <= upper)]

X = df.drop(columns=['output'])
y = df['output']

# Scaling
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Train-validation split for permutation importance
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Compute permutation importance using fixed stack model
fixed_estimators = [
    ('lr', LinearRegression()),
    ('ridge', Ridge()),
    ('bayes', BayesianRidge()),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
    ('gb', GradientBoostingRegressor()),
    ('xgb', XGBRegressor(verbosity=0)),
    ('lgb', LGBMRegressor(verbose=-1, n_estimators=100, learning_rate=0.05)),
]

stack_model_full = StackingRegressor(
    estimators=fixed_estimators,
    final_estimator=Ridge(),
    passthrough=False,
    cv=5,
    n_jobs=-1
)
stack_model_full.fit(X_train_full, y_train_full)

perm_result = permutation_importance(
    stack_model_full, X_val_full, y_val_full, n_repeats=10, random_state=42, scoring='r2'
)
perm_importance_stack = pd.Series(perm_result.importances_mean, index=X.columns).sort_values(ascending=False)
top_13_features = perm_importance_stack.head(13).index.tolist()

print("\n📊 Top 13 Important Features:")
print(top_13_features)

X_13 = X_scaled[top_13_features]

# 3. Optuna hyperparameter optimization
def objective(trial):
    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
        max_depth=trial.suggest_int("rf_max_depth", 3, 12),
        random_state=42
    )
    gb = GradientBoostingRegressor(
        n_estimators=trial.suggest_int("gb_n_estimators", 50, 300),
        learning_rate=trial.suggest_float("gb_lr", 0.01, 0.3),
        random_state=42
    )
    xgb = XGBRegressor(
        n_estimators=trial.suggest_int("xgb_n_estimators", 50, 300),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 12),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3),
        verbosity=0,
        random_state=42
    )
    lgb = LGBMRegressor(
        n_estimators=trial.suggest_int("lgb_n_estimators", 50, 300),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.3),
        verbose=-1,
        random_state=42
    )
    ridge_alpha = trial.suggest_float("ridge_alpha", 0.01, 10.0)

    estimators = [
        ('lr', LinearRegression()),
        ('ridge', Ridge()),
        ('bayes', BayesianRidge()),
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb),
        ('lgb', lgb),
    ]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ridge_alpha),
        passthrough=False,
        cv=5,
        n_jobs=-1
    )
    scores = cross_validate(
        stack, X_13, y, cv=5, scoring='r2', return_train_score=False, n_jobs=-1
    )
    return scores['test_score'].mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n🏆 Best Trial Hyperparameters:")
for key, val in study.best_trial.params.items():
    print(f"    {key}: {val}")

# 4. Train final model with best hyperparameters
params = study.best_trial.params

best_estimators = [
    ('lr', LinearRegression()),
    ('ridge', Ridge()),
    ('bayes', BayesianRidge()),
    ('rf', RandomForestRegressor(n_estimators=params['rf_n_estimators'], max_depth=params['rf_max_depth'], random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=params['gb_n_estimators'], learning_rate=params['gb_lr'], random_state=42)),
    ('xgb', XGBRegressor(n_estimators=params['xgb_n_estimators'], max_depth=params['xgb_max_depth'], learning_rate=params['xgb_lr'], verbosity=0, random_state=42)),
    ('lgb', LGBMRegressor(n_estimators=params['lgb_n_estimators'], learning_rate=params['lgb_lr'], verbose=-1, random_state=42)),
]

final_stack_model = StackingRegressor(
    estimators=best_estimators,
    final_estimator=Ridge(alpha=params['ridge_alpha']),
    passthrough=False,
    cv=5,
    n_jobs=-1
)
final_stack_model.fit(X_13, y)

# 5. Load and preprocess test data
test_df = pd.read_csv('test.csv')
test_ids = test_df['id']
test_df.drop(columns=['id', 'Row#'], inplace=True, errors='ignore')

X_test_scaled = pd.DataFrame(scaler.transform(test_df), columns=X.columns)
X_test_final = X_test_scaled[top_13_features]

# 6. Predict and save submission
y_pred_test = final_stack_model.predict(X_test_final)
submission_df = pd.DataFrame({
    'id': test_ids,
    'yield': y_pred_test
})
submission_df.to_csv('submission.csv', index=False)

print("\n✅ Submission file 'submission.csv' created with Optuna-tuned StackingRegressor.")
