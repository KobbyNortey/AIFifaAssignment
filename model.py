import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


fifa_22 = pd.read_csv("Data/players_22.csv", low_memory=False, dtype={25: str, 108: str})
male_players = pd.read_csv("Data/males_legacy.csv", low_memory=False, dtype={108: str})

#Cleaning
numeric_cols = male_players.select_dtypes(include=[np.number]).columns

# Unnecessary columns
drop_cols = ['club_contract_valid_until_year', 'fifa_update', 'fifa_version', 'league_id']
numeric_cols = numeric_cols.difference(drop_cols)


imputer = SimpleImputer(strategy='median')
male_players_filled = pd.DataFrame(imputer.fit_transform(male_players[numeric_cols]), columns=numeric_cols)

variances = male_players_filled.var()
non_zero_var_features = variances[variances > 1e-8].index.tolist()

male_players_filled = male_players_filled[non_zero_var_features]


y = male_players_filled['overall']
X = male_players_filled.drop(columns=['overall'])

selector = SelectKBest(f_regression, k=13)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()].tolist()
X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create models
knn_model = KNeighborsRegressor()
lr_model = LinearRegression()
svr_model = SVR()
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

# Train models with cross-validation
models = [knn_model, lr_model, svr_model, rf_model, gb_model]
model_names = ['KNN', 'Linear Regression', 'SVR', 'Random Forest', 'Gradient Boosting']

for model, name in zip(models, model_names):
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    print(f"{name} Cross-Validation RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")

best_model = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("Best parameters:", grid_search.best_params_)
print("Validation RMSE:", rmse)

numeric_cols_22 = fifa_22.select_dtypes(include=[np.number]).columns

# Rename sofifa_id to player_id in fifa_22 to align with male_players
fifa_22.rename(columns={'sofifa_id': 'player_id'}, inplace=True)

fifa_22_aligned = fifa_22[numeric_cols.intersection(fifa_22.columns)].copy()

fifa_22_filled = pd.DataFrame(imputer.transform(fifa_22_aligned), columns=fifa_22_aligned.columns)

X_test = fifa_22_filled[selected_features]

X_test_scaled = scaler.transform(X_test)

y_pred_test = best_model.predict(X_test_scaled)

rmse_test = np.sqrt(mean_squared_error(fifa_22_filled['overall'], y_pred_test))

print("Test RMSE:", rmse_test)


best_model = grid_search.best_estimator_

# Save the model and preprocessing objects
joblib.dump(best_model, "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")