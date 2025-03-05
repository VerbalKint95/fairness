## **Rapport d'Analyse de Fairness et de Modélisation Prédictive**

### **1. Introduction**

Ce projet vise à évaluer la fairness d'un modèle prédictif basé sur un jeu de données concernant les communautés et le crime. Plus précisément, il s'agit d'analyser et de prédire certaines caractéristiques des communautés, comme le taux de criminalité, en prenant en compte la race et d'autres attributs sensibles. L'objectif est non seulement de créer un modèle performant mais aussi de s'assurer que ce modèle est équitable pour tous les groupes protégés (notamment les groupes raciaux).

### **2. Description de la base de données**

Le jeu de données utilisé dans ce projet provient de l'étude des **Communities and Crime**. Ce dataset est constitué de diverses informations concernant les caractéristiques démographiques, économiques et sociales des communautés américaines, ainsi que des taux de criminalité. Les attributs clés incluent :

- **racepctblack** : Pourcentage de la population d'une commune identifiée comme noire.
- **racePctWhite** : Pourcentage de la population d'une commune identifiée comme blanche.
- **racePctAsian** : Pourcentage de la population d'une commune identifiée comme asiatique.
- **racePctHisp** : Pourcentage de la population d'une commune identifiée comme hispanique.
- **crime rate** : Taux de criminalité, qui sera prédictible via le modèle.
  
Chaque observation représente une commune différente, et ces attributs sont utilisés pour prédire les taux de criminalité, tout en prenant en compte les différences raciales dans les données.

#### **Attributs sensibles** :
Dans ce cas, les attributs sensibles sont principalement **racepctblack**, **racePctWhite**, **racePctAsian**, et **racePctHisp**, qui représentent les pourcentages de différentes races dans chaque communauté. Ces variables sont particulièrement intéressantes car elles peuvent avoir un impact sur la prédiction du taux de criminalité, créant ainsi des risques de biais dans le modèle.

#### **Cible (Label)** :
La variable cible du modèle est le taux de criminalité, que l'on cherche à prédire à partir des autres caractéristiques.

### **3. Méthodologie**

L'objectif principal de cette analyse est de construire un modèle prédictif de la criminalité tout en prenant en compte l’équité entre les différents groupes protégés. La méthodologie employée suit les étapes suivantes :

#### 3.1 **Prétraitement des données** :

- **Nettoyage des données** : Nous avons tout d'abord nettoyé le jeu de données en supprimant les valeurs manquantes et en standardisant les données.
  
- **Sélection des caractéristiques sensibles** : Nous avons défini les attributs sensibles comme étant les variables relatives aux races (racepctblack, racePctWhite, racePctAsian, racePctHisp).

- **Création des classes de privilège et de non-privilège** : Pour chaque groupe racial, un seuil a été défini pour déterminer si une communauté appartient au groupe "priviliégié" ou "non-priviliégié", basé sur un critère comme les pourcentages de la population (ex : une valeur inférieure à un certain seuil est considérée comme "non-priviliégiée").

#### 3.2 **Modélisation** :

- **Choix du modèle** : Nous avons utilisé un modèle de régression linéaire ou un modèle de régression logistique (en fonction de la nature de la cible) pour prédire le taux de criminalité. Ces modèles ont été entraînés en utilisant les caractéristiques démographiques, économiques et sociales des communautés.
  
- **Entraînement du modèle** : Le modèle a été entraîné en utilisant l'ensemble d'entraînement, tout en s'assurant que les biais raciaux étaient correctement gérés pendant le processus. Le modèle a été évalué à l'aide de critères de performance classiques tels que le RMSE et le R².

- **Évaluation de la fairness** : Une fois le modèle entraîné, il a été soumis à un ensemble de tests de fairness pour mesurer son impartialité. Ces tests incluent des métriques comme le **Disparate Impact**, la **Mean Difference**, et la **Statistical Parity Difference**.

### **4. Modélisation et Évaluation de la Fairness**

#### 4.1 **Évaluation des performances du modèle**

L'évaluation du modèle a été réalisée à l'aide des mesures de performance classiques pour les modèles de régression, à savoir :

- **RMSE (Root Mean Squared Error)** : 0.1276, ce qui indique que l'erreur moyenne quadratique du modèle est relativement faible et que le modèle prédit les taux de criminalité avec une précision décente.
- **R² (Coefficient de détermination)** : 0.6601, ce qui suggère que le modèle est capable d'expliquer environ 66% de la variance des taux de criminalité à partir des données d'entrée.

#### 4.2 **Évaluation de la fairness du modèle**

Les résultats des métriques de fairness montrent l'impact du modèle sur les groupes protégés. Voici un résumé des résultats obtenus pour chaque attribut racial :

##### 4.2.1 **Racepctblack** :
- **Disparate Impact** : 0.0342 (indiquant une grande disparité entre les groupes, où les groupes non-privilégiés sont défavorisés)
- **Mean Difference** : -0.0852 (différence moyenne entre les résultats pour les groupes protégés et non protégés)
- **Statistical Parity Difference** : -0.0852 (similaire à la Mean Difference, cela indique un déséquilibre systématique en faveur des groupes plus privilégiés)

##### 4.2.2 **RacePctWhite** :
- **Disparate Impact** : 24.0 (indique que le modèle est beaucoup plus favorable aux groupes blancs)
- **Mean Difference** : 0.1825 (les résultats sont plus favorables aux groupes blancs)
- **Statistical Parity Difference** : 0.1825

##### 4.2.3 **RacePctAsian** :
- **Disparate Impact** : inf (indique une division par zéro dans les calculs, ce qui signifie probablement qu'il n'y a pas de communautés asiatiques non-privilégiées dans l'échantillon de test)
- **Mean Difference** : 0.0206
- **Statistical Parity Difference** : 0.0206

##### 4.2.4 **RacePctHisp** :
- **Disparate Impact** : 0.1998
- **Mean Difference** : -0.0462
- **Statistical Parity Difference** : -0.0462

Ces résultats montrent des disparités importantes dans la prédiction des taux de criminalité, en particulier pour les groupes blancs et noirs. Le modèle favorise fortement les groupes blancs, tandis que les groupes noirs, hispaniques et asiatiques sont défavorisés.

### **5. Conclusion et Recommandations**

L’analyse des performances et des résultats de fairness montre que bien que le modèle soit relativement précis dans ses prédictions, il présente des biais significatifs en ce qui concerne les groupes raciaux. Pour atténuer ces biais, plusieurs approches de **dé-biaisage** peuvent être mises en œuvre, telles que :
1. L'application de techniques de pré-traitement comme le rééchantillonnage et la pondération des instances pour équilibrer la représentation des groupes dans les données d'entraînement.
2. L'utilisation d'approches d'entraînement équitables (in-processing) comme l'**adversarial debiasing** pour encourager le modèle à ignorer les attributs sensibles.
3. L'application de techniques de post-traitement pour ajuster les prédictions du modèle afin d'assurer l'équité entre les groupes.

L'application de ces méthodes pourrait améliorer significativement l’équité du modèle tout en maintenant une bonne performance prédictive.
