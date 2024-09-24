import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Fonction pour charger le fichier
def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.error("Veuillez télécharger un fichier CSV ou Excel valide.")
                return None
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return None
        return df
    else:
        st.error("Aucun fichier n'a été téléchargé.")
        return None

# Fonction pour convertir les colonnes datetime en valeurs utilisables
def convert_datetime_columns(df):
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df[col+'_year'] = df[col].dt.year
        df[col+'_month'] = df[col].dt.month
        df[col+'_day'] = df[col].dt.day
    df = df.drop(columns=datetime_cols)  # Supprimer les colonnes datetime originales
    return df

# Fonction pour afficher la heatmap de corrélation
def show_heatmap(df):
    st.subheader("Matrice de corrélation (Heatmap)")

    # Encoder les colonnes catégorielles avec One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)  # Convertit les colonnes catégorielles en colonnes numériques

    # Calculer la matrice de corrélation
    corr = df_encoded.corr()

    # Afficher la heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

# Modèle de Random Forest pour la prédiction
def train_random_forest(df, target_column):
    # Conversion des colonnes datetime
    df = convert_datetime_columns(df)

    # Séparation des features (X) et de la cible (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encoder les colonnes catégorielles avec One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)

    # Remplir les valeurs manquantes pour les colonnes numériques uniquement
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
    
    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardisation des données (Seulement pour les colonnes numériques)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialiser et entraîner le modèle Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédiction sur les données de test
    y_pred = model.predict(X_test)

    # Calcul de l'erreur (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse, X_train, X_test, y_test, y_pred

# Interface utilisateur simplifiée pour la prédiction
def show_prediction_interface(df):
    st.subheader("Prédiction des ventes")
    
    # Vérifier si les colonnes 'date' et 'produit' existent, puis les exclure de la sélection
    excluded_columns = ['date', 'produit']
    
    # Convert all columns to lowercase for consistent exclusion
    df.columns = df.columns.str.lower()
    
    # Check for existing columns in df to exclude
    selectable_columns = [col for col in df.columns if col not in excluded_columns]

    st.write(f"Colonnes disponibles après exclusion: {selectable_columns}")  # Debugging step

    # Si aucune colonne n'est disponible pour la prédiction après exclusion
    if len(selectable_columns) == 0:
        st.error("Aucune colonne valide disponible pour la prédiction après exclusion de 'date' et 'produit'.")
        return

    target_column = st.selectbox(
        "Sélectionnez la colonne cible (Ventes) à prédire :",
        selectable_columns
    )

    # Bouton pour entraîner le modèle et afficher les résultats
    if st.button("Lancer la prédiction"):
        model, mse, X_train, X_test, y_test, y_pred = train_random_forest(df, target_column)
        st.success(f"Modèle entraîné avec succès ! L'erreur moyenne quadratique (MSE) est de : {mse:.2f}")

        # Explication sur les résultats de la prédiction
        st.write("""
            **Interprétation de l'erreur quadratique moyenne (MSE)** :
            - Le MSE mesure l'écart moyen entre les valeurs prédites et les valeurs réelles.
            - Une valeur basse indique que les prédictions sont proches des valeurs réelles.
            - Un MSE élevé peut suggérer que le modèle a besoin d'améliorations ou que les données sont trop complexes.
        """)

        # Affichage des résultats de la prédiction
        results_df = pd.DataFrame({"Valeurs réelles": y_test, "Prédictions": y_pred})
        st.write("Voici les résultats des prédictions :")
        st.write(results_df)

        # Explication des résultats
        st.write("""
            **Interprétation des prédictions :**
            - Les 'Valeurs réelles' représentent les ventes réelles dans vos données.
            - Les 'Prédictions' sont les ventes prédites par le modèle Random Forest.
            - Une grande différence entre les valeurs réelles et les prédictions indique un ajustement imparfait.
        """)

        # Prédiction sur de nouvelles données (optionnel)
        st.subheader("Prédire de nouvelles données")
        file_new_data = st.file_uploader("Téléchargez un fichier de nouvelles données pour la prédiction", type=['csv', 'xlsx'])
        
        if file_new_data is not None:
            new_data = load_data(file_new_data)
            if new_data is not None:
                # Conversion des colonnes datetime dans les nouvelles données
                new_data = convert_datetime_columns(new_data)

                # Gérer les valeurs manquantes et l'encodage pour les nouvelles données
                new_data_encoded = pd.get_dummies(new_data, drop_first=True)
                missing_cols = set(X_train.columns) - set(new_data_encoded.columns)
                for col in missing_cols:
                    new_data_encoded[col] = 0
                
                # Standardiser les nouvelles données
                scaler = StandardScaler()
                new_data_scaled = scaler.fit_transform(new_data_encoded)

                # Effectuer la prédiction
                predictions = model.predict(new_data_scaled)
                st.write("**Résultats de la prédiction sur les nouvelles données :**")
                st.write(predictions)

                # Explication des résultats
                st.write("""
                    **Interprétation des prédictions sur de nouvelles données :**
                    - Les valeurs affichées sont les prédictions effectuées par le modèle pour les nouvelles données.
                    - Ces résultats représentent une estimation des ventes futures en fonction des caractéristiques fournies.
                """)

# Fonction principale pour l'upload et l'affichage
def main():
    st.title("📊 Dashboard de Prédiction des Ventes")
    st.write("Ce tableau de bord utilise un modèle Random Forest pour prédire les ventes à partir de vos données.")

    # Téléchargement du fichier
    file = st.sidebar.file_uploader("Téléchargez un fichier CSV ou Excel", type=['csv', 'xlsx'])
    df = load_data(file)

    if df is not None:
        # Affichage du tableau de bord de prédiction
        show_heatmap(df)
        show_prediction_interface(df)

if __name__ == "__main__":
    main()
