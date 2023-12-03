import pandas as pd

# Charger les données
data = pd.read_csv('data.csv')

# Afficher les premières lignes pour comprendre la structure des données
print(data.head())

# Vérifier la présence de valeurs manquantes
missing_values = data.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values)

# Imputation des valeurs manquantes
numeric_columns = data.select_dtypes(include=['number']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Imputation des valeurs manquantes pour les colonnes numériques
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Imputation des valeurs manquantes pour les colonnes catégorielles
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

# Vérifier les types de données
data_types = data.dtypes
print("Types de données par colonne :\n", data_types)

# Convertir des colonnes si nécessaire
# data['date_column'] = pd.to_datetime(data['date_column'])

# Supprimer les duplicatas
data = data.drop_duplicates()

# Vérifier les duplicatas après suppression
duplicates_count = data.duplicated().sum()
print("Nombre de duplicatas supprimés :", duplicates_count)

# Enregistrer les données nettoyées dans un nouveau fichier CSV
data.to_csv('clear_data.csv', index=False)
