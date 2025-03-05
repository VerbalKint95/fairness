import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data():
    communities_and_crime = fetch_ucirepo(id=183)
    X = communities_and_crime.data.features
    y = communities_and_crime.data.targets
    return X, y

def check_missing_values(X):
    missing_values = X.isnull().sum()
    print("Valeurs manquantes par attribut:")
    print(missing_values[missing_values > 0])

def drop_unwanted_columns(X):
    # Liste des colonnes à supprimer (en fonction des colonnes non pertinentes)
    columns_to_drop = ['state', 'county', 'community', 'communityname', 'fold']
    
    # Supprimer ces colonnes du DataFrame X
    X = X.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' pour ignorer si une colonne est déjà absente
    print(f"Colonnes supprimées : {columns_to_drop}")
    return X

def extract_sensitive_attributes(X):
    """Stocke les attributs sensibles pour analyse future, sans les retirer de X."""
    sensitive_columns = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    return X[sensitive_columns].copy()


def convert_columns_to_numeric(X):
    # Convertir les colonnes de type 'object' en 'float64'
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convertir la colonne en numérique (en remplaçant les erreurs par NaN)
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X

def impute_missing_values(X):
    # Calcul du pourcentage de données manquantes par attribut
    missing_values = X.isnull().sum()
    missing_percentage = (missing_values / len(X)) * 100
    
    # Supprimer les colonnes où plus de 50% des valeurs sont manquantes
    threshold = 50  # Seuil de 50% de données manquantes pour suppression
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    X = X.drop(columns=columns_to_drop)

    # Séparer les colonnes numériques et non numériques
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_columns = X.select_dtypes(exclude=['float64', 'int64']).columns

    # Imputer les colonnes numériques (moyenne)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X[numeric_columns])

    return X

def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_pca(X_scaled, n_components=10):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print("Variance expliquée par chaque composante principale :")
    print(pca.explained_variance_ratio_)
    return X_pca

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Chargement des données
    X, y = load_data()

    # Prétraitement des données
        
    check_missing_values(X)

    # Supprimer les colonnes inutiles
    X = drop_unwanted_columns(X)  

    # Extraction des variables sensibles (sans les supprimer)
    X_sensitive = extract_sensitive_attributes(X)

    # Conversion des colonnes 'object' en numériques
    X = convert_columns_to_numeric(X)

    # Imputation des valeurs manquantes
    X_imputed = impute_missing_values(X)
    
    # Standardisation des données
    X_scaled = standardize_data(X_imputed)

    # Application de la PCA
    X_pca = apply_pca(X_scaled, n_components=10)

    # Séparation des données
    X_train, X_test, y_train, y_test = split_data(X_pca, y)
    #X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    X_sensitive_train, X_sensitive_test, _, _ = split_data(X_sensitive, y)

    print("Dimensions des données d'entraînement et de test:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    print("Dimensions des attributs sensibles stockés pour fairness:")
    print(f"X_sensitive_train: {X_sensitive_train.shape}")
    print(f"X_sensitive_test: {X_sensitive_test.shape}")

    # Sauvegarde des données prétraitées
    np.save("data/X_train_pca.npy", X_train)
    np.save("data/X_test_pca.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)


    # Sauvegarde des noms des colonnes sensibles (pour réutilisation ultérieure)
    np.save("data/sensitive_columns.npy", X_sensitive.columns)

    np.save("data/X_sensitive_train.npy", X_sensitive_train)
    np.save("data/X_sensitive_test.npy", X_sensitive_test)

    print("✅ Données prétraitées enregistrées !")