from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

def train_models(df):
    X = df.drop('Rating', axis=1)
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({
            'Model': name,
            'R2': round(r2_score(y_test, preds), 3),
            'MAE': round(mean_absolute_error(y_test, preds), 3),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 3)
        })

    # Save the best model
    best_model = models['Random Forest']
    joblib.dump(best_model, "movie_rating_model.pkl")

    return results, best_model, X
if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess_data

    # Load and preprocess data
    path = "D:\\Dev\\Movie Rating pREDICTION\\IMDb_Movies_India.csv"
    df = preprocess_data(path)

    print("✅ Data Preprocessing Complete!")
    print("Shape:", df.shape)

    # Train models
    results, best_model, features = train_models(df)
    
    print("🚀 Model Evaluation:")
    for result in results:
        print(f"🔹 {result['Model']}: R²={result['R2']}, MAE={result['MAE']}, RMSE={result['RMSE']}")
    
    print("✅ Model training complete and saved as 'movie_rating_model.pkl'")
