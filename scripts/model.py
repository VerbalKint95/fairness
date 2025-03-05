import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import joblib


#-----EVALUATION----------------------------------------
def evaluate_predictionl(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"➡ RMSE: {rmse:.4f}")
    print(f"➡ R²: {r2:.4f}")


def  evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Évalue le modèle en affichant le RMSE et le R².
    """

    # Prédictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print("=" * 40)
    print(f"Performance du modèle {model_name}")
    print("")
    print("sur test set:")
    evaluate_predictionl(y_test, y_pred_test)
    print("")
    print("sur train set:")
    evaluate_predictionl(y_train, y_pred_train)
    print("=" * 40)

    


#-----MODELS----------------------------------------------
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Entraînement d'un modèle RandomForestRegressor.
    """
    # Création du modèle RandomForest
    model = RandomForestRegressor(
        n_estimators=200,       # Plus d'arbres pour la stabilité
        max_depth=10,           # Limite la profondeur pour éviter le sur-apprentissage
        min_samples_split=5,    # Empêche des divisions trop fines
        min_samples_leaf=4,     # Assure que les feuilles contiennent assez d'exemples
        max_features='sqrt',    # Prend une fraction des features à chaque split pour diversifier
        random_state=42
    )


    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Évaluation du modèle
    evaluate_model(model, "RandomForest", X_train, y_train, X_test, y_test)

    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Entraînement d'un modèle XGBoostRegressor avec des hyperparamètres optimisés.
    """
    # Initialiser le modèle XGBoost avec d'autres hyperparamètres
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=10000,               # Nombre d'arbres plus élevé
        learning_rate=0.01,             # Réduire le taux d'apprentissage
        max_depth=9,                    # Augmenter la profondeur de l'arbre
        random_state=42, 
        subsample=0.8,                  # Sous-échantillonnage des données
        colsample_bytree=0.8,           # Sous-échantillonnage des caractéristiques par arbre
        gamma=0.1,                      # Ajustement de la réduction minimale de la perte
        reg_lambda=1.0,                 # Regularisation L2
        reg_alpha=0.5                   # Regularisation L1
    )
    
    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Évaluation du modèle
    evaluate_model(model, "XGBoost", X_train, y_train, X_test, y_test)

    return model

def train_xgboost_with_random_search(X_train, y_train, X_test, y_test):
    """
    Entraînement d'un modèle XGBoost avec recherche d'hyperparamètres optimisée (RandomizedSearchCV).
    """
    param_grid = {
        'max_depth': [3, 4, 5],  # réduire la recherche pour gagner du temps
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 1.0]
    }

    # Initialisation du modèle
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Recherche aléatoire avec progression
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=10000,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Effectuer la recherche aléatoire
    random_search.fit(X_train, y_train)

    # Meilleurs hyperparamètres
    print(f"✅ Meilleurs paramètres : {random_search.best_params_}")

    # Modèle optimisé
    best_model = random_search.best_estimator_

    # Évaluation sur test et train
    evaluate_model(best_model,"XGBoost - RandomSearch",X_train, y_train, X_test, y_test)

    return best_model

def main():
    # Charger les données
    X_train = np.load("data/X_train_pca.npy")
    X_test = np.load("data/X_test_pca.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    
    # Exemple d'appel à train_xgboost avec recherche par grille
    print("Entraînement du modèle")
    model_xgb = train_xgboost_with_random_search(X_train, y_train, X_test, y_test)


    # Sauvegarde du modèle
    joblib.dump(model_xgb, 'models/best_xgboost_model.pkl')
    print("📁 Modèle sauvegardé : 'models/best_xgboost_model.pkl'")


if __name__ == "__main__":
    main()
