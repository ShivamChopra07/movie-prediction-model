import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title("🎯 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

def plot_model_comparison(results_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(results_df)
    df_melt = df.melt(id_vars="Model", value_vars=["R2", "MAE", "RMSE"])

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melt, x="Model", y="value", hue="variable")
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
    if __name__ == "__main__":
        # Example usage
        results = [
            {'Model': 'Linear Regression', 'R2': 0.85, 'MAE': 0.5, 'RMSE': 0.6},
            {'Model': 'Random Forest', 'R2': 0.9, 'MAE': 0.4, 'RMSE': 0.5}
        ]
        plot_model_comparison(results)
        print("✅ Visualization Complete!")