import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression

def get_data(path):
    """Load and clean air quality data from a file."""
    print("\n--- Loading & Cleaning Data ---\n")
    df = pd.read_csv(path)
    keys = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)', 'PT08.S5(O3)']
    print("Columns:", list(df.columns))
    for k in keys:
        if k not in df.columns:
            raise KeyError(f"Missing {k}")
        df[k] = pd.to_numeric(df[k], errors='coerce').replace(-200, np.nan)
    pre_na = df[keys].isna().sum()
    print("\nMissing (initial):\n", pre_na)
    df['dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce', dayfirst=True)
    df = df.dropna(subset=['dt'])
    bef = len(df)
    df = df.dropna(subset=keys, thresh=3)
    print(f"Dropped rows: {bef-len(df)}")
    df[keys] = df[keys].ffill().bfill()
    post_na = df[keys].isna().sum()
    print("\nMissing (after fill):\n", post_na)
    print("\nClean sample:\n", df[keys].head())
    return df, keys

def drop_outliers(df, keys):
    """Remove outliers using IQR for each pollutant, print each on one line."""
    print("\n\n--- Outlier Removal with IQR ---\n")
    for k in keys:
        q1 = df[k].quantile(0.25)
        q3 = df[k].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr
        before = len(df)
        df = df[(df[k] >= low) & (df[k] <= up)]
        removed = before - len(df)
        print(f"{k}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, low={low:.2f}, high={up:.2f}, removed={removed}")
    print(f"Rows after outlier removal: {len(df)}")
    return df


def norm_data(df, keys, mode='minmax'):
    """Scale features for comparison."""
    print("\n\n--- Normalizing Data ---")
    scaler = MinMaxScaler() if mode == 'minmax' else StandardScaler()
    print("\nBefore min/max:\n", df[keys].agg(['min','max']).T)
    df[keys] = scaler.fit_transform(df[keys])
    print("\nAfter min/max:\n", df[keys].agg(['min','max']).T)
    return df

def do_clustering(df, feats, n=3):
    """Make clusters and label data."""
    print("\n\n--- Clustering ---")
    model = KMeans(n_clusters=n, random_state=42, n_init=10)
    df['group'] = model.fit_predict(df[feats])
    print("\nGroups (by count):\n", pd.Series(df['group']).value_counts())
    print("\nCenters:\n", pd.DataFrame(model.cluster_centers_, columns=feats))
    print("\nSample rows:\n", df[feats+['group']].head())
    return df

def pick_features(df, pool, label, cut=0.01):
    """Select best features by information gain."""
    print("\n\n--- Selecting Features ---\n")
    clean = df.dropna(subset=[label])
    X = clean[pool].fillna(clean[pool].mean())
    y = clean[label]
    info = mutual_info_regression(X, y)
    table = pd.DataFrame({'feature': pool, 'gain': info}).sort_values(by='gain', ascending=False)
    print(table)
    toss = table[table['gain'] < cut]['feature'].tolist()
    print("\nDropping:", toss)
    keep = table[table['gain'] >= cut]['feature'].tolist()
    print("\nKeeping:", keep)
    return keep, table

def quick_viz(df, model, keys):
    print("\n\n--- Quick Data Viz ---")
    plt.figure(figsize=(7,5))
    sns.heatmap(df[keys].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
    df[keys].hist(bins=30, figsize=(10,6), color='skyblue')
    plt.suptitle("Distributions After Scaling")
    plt.show()
    plt.figure(figsize=(7,4))
    sns.histplot(df['stress_score'], bins=30, kde=True, color='coral')
    plt.title("Stress Score Histogram")
    plt.show()
    plt.figure(figsize=(8,4))
    sns.lineplot(x='hr', y='stress_score', data=df, marker='o', color='purple')
    plt.title("Avg Stress by Hour")
    plt.show()
    plt.figure(figsize=(7,5))
    sns.scatterplot(x='CO(GT)', y='NO2(GT)', hue='group', data=df, palette='viridis')
    plt.title("Clusters: CO vs NO2")
    plt.show()
    if model:
        plt.figure(figsize=(8,5))
        sns.barplot(x=model.feature_importances_, y=model.feature_names_in_, palette='magma')
        plt.title("Feature Importances")
        plt.show()

def main():
    print("\n\n                                                        =~=~=~= Air Quality Pipeline =~=~=~=\n\n")
    path = 'data/air_quality.csv'
    if not os.path.exists(path):
        raise FileNotFoundError("Missing air_quality.csv")
    df, keys = get_data(path)
    df = drop_outliers(df, keys)
    df = norm_data(df, keys)
    df['hr'] = df['dt'].dt.hour
    df['dy'] = df['dt'].dt.day
    df['mo'] = df['dt'].dt.month
    print("\nTime features added:\n", df[['hr','dy','mo']].head())
    weights = {'CO(GT)':1.5, 'NOx(GT)':2.5, 'NO2(GT)':2.0, 'C6H6(GT)':3.0, 'PT08.S5(O3)':2.0}
    df['stress_score'] = sum(df[k]*weights[k] for k in keys)
    print("\nStress score calculation:\n", df[['stress_score']+keys].head())
    df = do_clustering(df, keys + ['hr'], n=3)
    candidates = keys + ['hr','dy','mo']
    keep, info_table = pick_features(df, candidates, 'stress_score')
    model = None
    if keep:
        df_model = df.dropna(subset=keep + ['stress_score'])
        if not df_model.empty:
            X, y = df_model[keep], df_model['stress_score']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            print("\nRandom Forest trained.")
            for feat, imp in zip(keep, model.feature_importances_):
                print(f"{feat}: {imp:.3f}")
    quick_viz(df, model, keys)

if __name__ == "__main__":
    main()
