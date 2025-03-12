import pandas as pd

def load_data(route_id=2946091):
    # Charger les fichiers Excel
    customers_df = pd.read_excel('database/2_detail_table_customers.xls')
    depots_df = pd.read_excel('database/4_detail_table_depots.xls')
    distances_df = pd.read_excel('database/6_detail_table_cust_depots_distances.xls')

    # Filtrer les données pour le ROUTE_ID spécifié
    customers_df = customers_df[customers_df['ROUTE_ID'] == route_id]
    depots_df = depots_df[depots_df['ROUTE_ID'] == route_id]
    distances_df = distances_df[distances_df['ROUTE_ID'] == route_id]

    # Extraire les informations des clients
    customer_positions = list(zip(customers_df['CUSTOMER_LATITUDE'], customers_df['CUSTOMER_LONGITUDE']))
    orders = list(map(int, customers_df['NUMBER_OF_ARTICLES']))

    # Extraire les informations des dépôts
    depot_positions = list(zip(depots_df['DEPOT_LATITUDE'], depots_df['DEPOT_LONGITUDE']))

    # Extraire les distances
    distances = {}
    for _, row in distances_df.iterrows():
        key = (row['DEPOT_CODE'], row['CUSTOMER_CODE'])
        distances[key] = row['DISTANCE_KM']

    # Créer la structure de données
    state = {
        "position": [depot_positions[0]] + customer_positions,  # Ajouter le dépôt en premier
        "orders": [0] + orders  # La demande du dépôt est 0
    }

    return state #, distances
