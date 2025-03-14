import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import mean_squared_error, r2_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from xgboost import XGBRegressor


def isFavorable(value):
    return 1 if value < 0.50 else 0

def load_data():
    """
    Charger les données et appliquer la fonction is_privileged directement.
    """
    X_train = np.load("data/X_train_pca.npy")
    X_test = np.load("data/X_test_pca.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    # Attributs sensibles
    X_sensitive_train = np.load("data/X_sensitive_train.npy")
    X_sensitive_test = np.load("data/X_sensitive_test.npy")

    # Charger les noms des colonnes sensibles sauvegardées
    sensitive_columns = np.load("data/sensitive_columns.npy", allow_pickle=True)

    # Convertir en DataFrame
    X_sensitive_train_df = pd.DataFrame(X_sensitive_train, columns=sensitive_columns)
    X_sensitive_test_df = pd.DataFrame(X_sensitive_test, columns=sensitive_columns)

    def is_privileged(value):
        """
        Fonction pour déterminer si un pourcentage donné est dans le groupe privilégié
        On considère privilégié si le pourcentage est inférieur à 30%, sinon non privilégié.
        """
        return 1 if value < 0.30 else 0

    # Appliquer is_privileged à chaque colonne sensible
    X_sensitive_train_df = X_sensitive_train_df.map(is_privileged)
    X_sensitive_test_df = X_sensitive_test_df.map(is_privileged)

    return X_train, X_test, y_train, y_test, X_sensitive_train_df, X_sensitive_test_df

def load_model(model_name):
    """
    Charger le meilleur modèle XGBoost.
    """
    model = joblib.load('models/'+model_name+'.pkl')
    return model

def evaluate_prediction(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"➡ RMSE: {rmse:.4f}")
    print(f"➡ R²: {r2:.4f}")

def evaluate_model(model, model_name, X_test, y_test, X_train=None, y_train=None):
    """
    Évalue le modèle en affichant le RMSE et le R².
    """

    # Prédictions
    y_pred_test = model.predict(X_test)
    
    print("=" * 40)
    print(f"Performance du modèle {model_name}")
    print("")
    print("sur test set:")
    evaluate_prediction(y_test, y_pred_test)
    print("")
    if (X_train != None) and (y_train != None):
        y_pred_train = model.predict(X_train)
        print("sur train set:")
        evaluate_prediction(y_train, y_pred_train)
        print("=" * 40)

def evaluate_fairness(y_pred, X_sensitive_test, model_name, y_test):
    """
    Calculer les métriques de fairness pour plusieurs attributs sensibles.
    """

    # Convertir les prédictions et les étiquettes en DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=['predictions'])

    # Convertir les probabilités en classes binaires (0 ou 1)
    y_pred_df['predictions'] = y_pred_df['predictions'].map(isFavorable)
    
    # Convertir y_test (qui est aussi une probabilité) en classe binaire
    y_test_df = pd.DataFrame(y_test, columns=['true_labels'])
    y_test_df['true_labels'] = y_test_df['true_labels'].map(isFavorable)

    # Calculer l'accuracy
    accuracy = np.mean(y_pred_df['predictions'] == y_test_df['true_labels'])
    print(f"Accuracy: {accuracy:.4f}")

    # Créer un DataFrame combiné avec les prédictions, étiquettes et attributs sensibles
    combined_df = pd.concat([y_pred_df, pd.DataFrame(X_sensitive_test)], axis=1)
    
    # Liste des attributs sensibles
    sensitive_attributes = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    
    # Initialisation des listes pour stocker les métriques de fairness
    disparate_impact_values = []
    mean_diff_values = []
    stat_parity_diff_values = []

    for sensitive_attribute in sensitive_attributes:

        # Créer un dataset AIF360 (StandardDataset)
        dataset = StandardDataset(
            combined_df,                                    # dataframe
            label_name='predictions',                       # colonne avec les bons résultats
            favorable_classes=[1],                          # condition pour que le résultat soit favorable
            protected_attribute_names=[sensitive_attribute],# colonne protégée
            privileged_classes=[[1]],                       # condition pour être privilégié
        )

        # Calcul des métriques de fairness
        metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{sensitive_attribute: 1}], unprivileged_groups=[{sensitive_attribute: 0}])

        # Stocker les résultats des métriques de fairness
        disparate_impact = metric.disparate_impact()
        if np.isinf(disparate_impact):
            disparate_impact = 0
        mean_difference = metric.mean_difference()
        stat_parity_difference = metric.statistical_parity_difference()

        disparate_impact_values.append(disparate_impact)
        mean_diff_values.append(mean_difference)
        stat_parity_diff_values.append(stat_parity_difference)

        # Afficher les résultats des métriques de fairness
        print(f"Fairness Metrics pour l'attribut {sensitive_attribute}:")
        print(f"Disparate Impact: {disparate_impact}")
        print(f"Mean Difference: {mean_difference}")
        print(f"Statistical Parity Difference: {stat_parity_difference}")
        print("-" * 50)
    
    # Création des graphiques

    # 1. Disparate Impact
    plt.figure(figsize=(8, 6))
    plt.bar(sensitive_attributes, disparate_impact_values, color=['red', 'blue', 'green', 'orange'])
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=1, label='Seuil inférieur (0.8)')
    plt.axhline(y=1.25, color='red', linestyle='--', linewidth=1, label='Seuil supérieur (1.25)')
    plt.axhline(y=1, color='black', linestyle='-', linewidth=1, label='Équité parfaite (1)')
    plt.title('Disparate Impact par attribut sensible')
    plt.xlabel('Attribut sensible')
    plt.ylabel('Disparate Impact')
    plt.ylim(0, 2.5)  # Ajuster l'axe Y pour avoir une échelle logique
    plt.savefig('saves/'+model_name+'_disparate_impact.png')
    plt.close()

    # 2. Mean Difference
    plt.figure(figsize=(8, 6))
    plt.bar(sensitive_attributes, mean_diff_values, color=['red', 'blue', 'green', 'orange'])
    plt.title('Mean Difference par attribut sensible')
    plt.xlabel('Attribut sensible')
    plt.ylabel('Mean Difference')
    plt.ylim(-0.6, 0.6)  # Ajuster l'axe Y pour une meilleure visualisation
    plt.savefig('saves/'+model_name+'_mean_difference.png')
    plt.close()

    # 3. Statistical Parity Difference
    plt.figure(figsize=(8, 6))
    plt.bar(sensitive_attributes, stat_parity_diff_values, color=['red', 'blue', 'green', 'orange'])
    plt.title('Statistical Parity Difference par attribut sensible')
    plt.xlabel('Attribut sensible')
    plt.ylabel('Statistical Parity Difference')
    plt.ylim(-0.6, 0.6)  # Ajuster l'axe Y
    plt.savefig('saves/'+model_name+'_statistical_parity_difference.png')
    plt.close()

    print("Les graphiques ont été enregistrés dans le dossier 'save/'.")

def debias_by_reweight_and_train(X_train, y_train, X_sensitive_train):

    # Convertir en DataFrame pour AIF360
    y_train_df = pd.DataFrame(y_train, columns=['true_labels'])


    y_train_df = y_train_df.map(isFavorable)

    combined_train_df = pd.concat([y_train_df, X_sensitive_train], axis=1)

    dataset_train = StandardDataset(
        combined_train_df,
        label_name='true_labels',
        favorable_classes=[1],
        protected_attribute_names=['racepctblack', 'racePctWhite', 'racePctHisp'],
        privileged_classes=[[1]]
    )

    reweigher = Reweighing(
        unprivileged_groups=[{'racepctblack': 0}, {'racePctHisp': 0}, {'racePctWhite': 1}],
        privileged_groups=[{'racepctblack': 1}, {'racePctHisp': 1}, {'racePctWhite': 0}]
    )

    debiased_train_dataset = reweigher.fit_transform(dataset_train)
    sample_weights = debiased_train_dataset.instance_weights

    debiased_model = XGBRegressor(objective="reg:squarederror")
    debiased_model.fit(X_train, y_train, sample_weight=sample_weights)

    joblib.dump(debiased_model, 'models/debiased_xgboost_model.pkl')
    return debiased_model

def postprocess_fairness(dataset, y_pred):
    eq_odds = CalibratedEqOddsPostprocessing(
        privileged_groups=[{'racepctblack': 1}],
        unprivileged_groups=[{'racepctblack': 0}],
        seed=42
    )
    y_pred_binary = np.vectorize(isFavorable)(y_pred)
    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred_binary
    dataset_transformed = eq_odds.fit_predict(dataset, dataset_pred)
    return dataset_transformed.labels

def main():
    # Charger les données
    X_train, X_test, y_train, y_test, X_sensitive_train, X_sensitive_test = load_data()

    # Charger le meilleur modèle
    model = load_model("best_xgboost_model")
    
    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)
    
    # Évaluer la performance du modèle
    evaluate_model(model, "biased_model", X_test, y_test)

    # Évaluer la fairness du modèle
    print("Évaluation de la fairness du modèle :")
    evaluate_fairness(y_pred, X_sensitive_test, "original", y_test)

     # Appliquer le débiaisage et réentraîner le modèle
    debiased_model = debias_by_reweight_and_train(X_train, y_train, X_sensitive_train)
    y_pred_debiased = debiased_model.predict(X_test)

    # Évaluer la performance du modèle après reweight
    evaluate_model(debiased_model, "debiased_model", X_test, y_test)
    evaluate_fairness(y_pred_debiased, X_sensitive_test, "debiased", y_test)
    
    # Évaluer la performance du modèle après post process
    dataset_test = StandardDataset(pd.concat([pd.DataFrame(y_test, columns=['true_labels']), X_sensitive_test], axis=1),
                                   label_name='true_labels',
                                   favorable_classes=[1],
                                   protected_attribute_names=['racepctblack', 'racePctWhite', 'racePctHisp'],
                                   privileged_classes=[[1]])
    y_pred_postprocessed = postprocess_fairness(dataset_test, y_pred_debiased)
    evaluate_fairness(y_pred_postprocessed, X_sensitive_test, "postprocessed", y_test)

if __name__ == "__main__":
    main()
