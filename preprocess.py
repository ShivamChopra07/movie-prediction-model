import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path, encoding='latin1')

    # Drop rows with missing rating
    df = df.dropna(subset=['Rating'])
    df.fillna('Unknown', inplace=True)

    df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Votes', 'Rating']]

    # Genre dummies
    genre_dummies = df['Genre'].str.get_dummies(sep=',')
    df = pd.concat([df.drop('Genre', axis=1), genre_dummies], axis=1)

    # Encode Director & Actors
    le_director = LabelEncoder()
    df['Director'] = le_director.fit_transform(df['Director'])

    le1, le2, le3 = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Actor 1'] = le1.fit_transform(df['Actor 1'])
    df['Actor 2'] = le2.fit_transform(df['Actor 2'])
    df['Actor 3'] = le3.fit_transform(df['Actor 3'])

    # Combine actors into single column
    df['actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].mean(axis=1)
    df.drop(['Actor 1', 'Actor 2', 'Actor 3'], axis=1, inplace=True)

    # Clean and scale votes
    df['Votes'] = df['Votes'].str.replace(',', '')
    df['Votes'] = df['Votes'].astype(int)
    scaler = StandardScaler()
    df['Votes'] = scaler.fit_transform(df[['Votes']])

    return df
if __name__ == "__main__":
    # Example usage
    path = "D:\\Dev\\Movie Rating pREDICTION\\IMDb_Movies_India.csv"
    processed_df = preprocess_data(path)
    print("✅ Data Preprocessing Complete!")
    print("Shape:", processed_df.shape)
    print(processed_df.head())
