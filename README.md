# 🏠 Prédiction des loyers immobiliers en France

📊 Web scraping, modélisation prédictive et interprétabilité des loyers en France  
à l’aide de modèles de Machine Learning et d’une application **Streamlit**.

---

## 📌 Présentation

Ce projet met en œuvre un **pipeline complet de data science** permettant de :

- 🌐 Collecter automatiquement des annonces immobilières
- 🧹 Nettoyer et préparer les données
- 📈 Analyser le marché locatif français
- 🔮 Prédire le loyer mensuel d’un bien immobilier
- 🔍 Interpréter les résultats des modèles

Les modèles utilisés sont :
- Régression linéaire
- Random Forest

---

## ⚠️ Avertissement important — Temps de scraping

🚨 **À lire avant toute exécution**

Le scraping des annonces immobilières est **entièrement synchrone**.

⏱️ **Durée estimée pour un scraping complet : 14 à 15 heures**

❌ Il est fortement déconseillé de lancer le scraping complet pour un premier test.

✅ Une **base de données déjà scrapée (18/11/2025)** est fournie avec le projet  
et permet une **utilisation immédiate**.

---

## 🗂️ Structure du projet

.
├── RentalScraper.py  
│   └── Web scraping des annonces immobilières  

├── RentalCleaner.py  
│   └── Nettoyage et préparation des données  

├── RentalStatsViews.py  
│   └── Analyse exploratoire et visualisations  

├── RentalRegression.py  
│   └── Modélisation, évaluation et interprétabilité (SHAP)  

├── main.py  
│   └── Application Streamlit (orchestration du pipeline)  

├── scraping_outputs/  
│   └── rental_database.parquet  

├── logs/  
│   └── Fichiers de logs  

└── README.md  

---

## 🔍 Fonctionnalités

- 🌐 Web scraping robuste (gestion des erreurs, retry, logging)
- 🧹 Nettoyage avancé basé sur des règles métier réelles
- 🧠 Feature engineering (prix au m², transformations logarithmiques)
- 📊 Analyse exploratoire des données (EDA)
- 🤖 Modélisation :
  - Régression linéaire
  - Random Forest
- 📐 Validation des modèles :
  - Train / Test
  - Cross-validation
  - MAE, RMSE, R²
- 📉 Analyse des résidus
- 🔎 Interprétabilité :
  - Permutation Feature Importance
  - Valeurs SHAP
- 🖥️ Application interactive Streamlit

---

## ▶️ Utilisation rapide

La base de données étant déjà fournie, il suffit de lancer l’application Streamlit.

Commande :

streamlit run main.py

Puis ouvrir le navigateur à l’adresse :

http://localhost:8501

---

## 🔁 Relancer le scraping (optionnel)

Pour relancer la collecte des données :

python RentalScraper.py

📂 La base sera automatiquement sauvegardée dans :

scraping_outputs/rental_database.parquet

---

## 🧪 Mode test du scraping (recommandé)

Pour tester le scraping sans attendre plusieurs heures :

1️⃣ Ouvrir le fichier `RentalScraper.py`  
2️⃣ Dans la méthode `get_url_suffixes`, définir :

rent_step = 40000  
sped_up_rent_step = 40000  

⚠️ Ce mode est réservé aux tests techniques.  
La base générée n’est **pas représentative** du marché réel.

---

## 📊 Données

📍 Source : locamoi.fr  

📦 Environ 94 000 annonces collectées  
📦 Environ 80 000 observations exploitables après nettoyage  
🗺️ Couverture nationale (France métropolitaine + DOM)

---

## 🛠️ Technologies utilisées

- 🐍 Python
- 🌐 requests
- 🍜 BeautifulSoup
- 🧮 pandas, numpy
- 🤖 scikit-learn
- 📊 matplotlib, seaborn
- 🔍 shap
- 🖥️ Streamlit

---

## 👤 Auteur

👨‍💻 **Paulo Sergio Garcia Rodriguez**  
🎓 Projet académique — Paris, 2025
