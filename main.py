import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Change the path to match your system
Data = pd.read_csv(r"D:\Dev\Movie Rating pREDICTION\IMDb_Movies_India.csv", encoding='latin1')

print("✅ Dataset Loaded!")
print("Shape:", Data.shape)
print(Data.head())

# Drop missing ratings
Data2 = Data.dropna(subset=['Rating'])

# Fill other missing values
Data2.fillna('Unknown', inplace=True)
print("✅ Missing Values Handled!")
print("Shape after dropping missing ratings:", Data2.shape)

# Keep important columns
Data3 = Data2[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Votes', 'Rating']]
# Convert genre to dummy columns
genre_dummies = Data3['Genre'].str.get_dummies(sep=',')
Data4 = pd.concat([Data3.drop('Genre', axis=1), genre_dummies], axis=1)

# Encode director
Director_label = LabelEncoder()
Data4['Director'] = Director_label.fit_transform(Data4['Director'])


# Encode Actor 1, 2, 3 separately
lebel1 = LabelEncoder()
lebel2 = LabelEncoder()
lebel3 = LabelEncoder()

Data4['Actor 1'] = lebel1.fit_transform(Data4['Actor 1'])
Data4['Actor 2'] = lebel2.fit_transform(Data4['Actor 2'])
Data4['Actor 3'] = lebel3.fit_transform(Data4['Actor 3'])

# Combine into one 'actors' column
Data4['actors'] = Data4[['Actor 1', 'Actor 2', 'Actor 3']].mean(axis=1)

# Drop old actor columns
Data4.drop(['Actor 1', 'Actor 2', 'Actor 3'], axis=1, inplace=True)

# Scale votes
# Clean and scale votes
Data4['Votes'] = Data4['Votes'].str.replace(',', '')     # Remove commas like '1,086' → '1086'
Data4['Votes'] = Data4['Votes'].astype(int)              # Convert string to int

scaler = StandardScaler()
Data4['Votes'] = scaler.fit_transform(Data4[['Votes']])  # Now scale it

#Split dataset into features and target variable
X = Data4.drop('Rating', axis=1)
y = Data4['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train Models
print("\n🚀 Model Evaluation:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"🔹 {name}")
    print("R² Score :", round(r2_score(y_test, predictions), 3))
    print("Mean Absolute Error      :", round(mean_absolute_error(y_test, predictions), 3))
    print("Root Mean Squared Error     :", round(np.sqrt(mean_squared_error(y_test, predictions)), 3))
    print("-" * 40)

#   FEATURE IMPORTANCE
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("🎯 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

#Saving The Model
joblib.dump(rf_model, "movie_rating_model.pkl")
print("✅ Model saved as 'movie_rating_model.pkl'")
