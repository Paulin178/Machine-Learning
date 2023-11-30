import pandas as pd

# Charger les données, entrez le nom de votre dataset entre les (parenthèses)
data = pd.read_csv('data.csv')

# Afficher les premières lignes pour comprendre la structure des données
print(data.head())

# Vérifier la présence de valeurs manquantes
missing_values = data.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values)

# Supprimer les lignes avec des valeurs manquantes si nécessaire
data = data.dropna()

# Supprimer les duplicatas
data = data.drop_duplicates()

# Vérifier les types de données
data_types = data.dtypes
print("Types de données par colonne :\n", data_types)


# Effectuer d'autres opérations de nettoyage spécifiques à votre ensemble de données

# Enregistrer les données nettoyées dans un nouveau fichier CSV
data.to_csv('clear_data.csv', index=False)
