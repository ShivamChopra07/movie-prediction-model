def add_actor_popularity(df):
    actor_freq = df['actors'].value_counts().to_dict()
    df['Actor_Popularity'] = df['actors'].map(actor_freq)
    return df

def add_director_hit_count(df):
    df['High_Rated'] = (df['Rating'] >= 7).astype(int)
    director_hits = df.groupby('Director')['High_Rated'].sum().to_dict()
    df['Director_Hit_Count'] = df['Director'].map(director_hits)
    df.drop('High_Rated', axis=1, inplace=True)
    return df
if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess_data

    # Load and preprocess data
    path = "D:\\Dev\\Movie Rating pREDICTION\\IMDb_Movies_India.csv"
    df = preprocess_data(path)

    print("✅ Data Preprocessing Complete!")
    print("Shape:", df.shape)

    # Add actor popularity and director hit count
    df = add_actor_popularity(df)
    df = add_director_hit_count(df)

    print("✅ Feature Engineering Complete!")
    print(df.head())
