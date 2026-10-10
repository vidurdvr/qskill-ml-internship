import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HousingFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to engineer domain-specific housing features:
    1. bedrooms_ratio: AveBedrms / AveRooms
    2. rooms_per_household: AveRooms
    3. population_per_household: AveOccup
    4. rooms_per_occupant: AveRooms / (AveOccup + 1e-6)
    5. distance_to_la: Euclidean proxy distance to Los Angeles center (34.0522, -118.2437)
    6. distance_to_sf: Euclidean proxy distance to San Francisco center (37.7749, -122.4194)
    """
    def __init__(self, include_geo_distances: bool = True):
        self.include_geo_distances = include_geo_distances
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Support both DataFrame and numpy array
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Assume original feature order
            cols = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])

        # 1. Ratio of bedrooms to total rooms
        df['bedrooms_ratio'] = df['AveBedrms'] / np.maximum(df['AveRooms'], 1e-5)
        
        # 2. Rooms per occupant
        df['rooms_per_occupant'] = df['AveRooms'] / np.maximum(df['AveOccup'], 1e-5)

        # 3. Geo distances to primary California urban economic hubs
        if self.include_geo_distances and 'Latitude' in df.columns and 'Longitude' in df.columns:
            la_coords = (34.0522, -118.2437)
            sf_coords = (37.7749, -122.4194)
            
            df['dist_to_la'] = np.sqrt(
                (df['Latitude'] - la_coords[0])**2 + (df['Longitude'] - la_coords[1])**2
            )
            df['dist_to_sf'] = np.sqrt(
                (df['Latitude'] - sf_coords[0])**2 + (df['Longitude'] - sf_coords[1])**2
            )
            df['min_dist_to_major_city'] = np.minimum(df['dist_to_la'], df['dist_to_sf'])

        self.feature_names_out_ = list(df.columns)
        return df.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function for raw pandas DataFrame analysis during EDA."""
    data = df.copy()
    data['bedrooms_ratio'] = data['AveBedrms'] / np.maximum(data['AveRooms'], 1e-5)
    data['rooms_per_occupant'] = data['AveRooms'] / np.maximum(data['AveOccup'], 1e-5)
    
    la_coords = (34.0522, -118.2437)
    sf_coords = (37.7749, -122.4194)
    data['dist_to_la'] = np.sqrt((data['Latitude'] - la_coords[0])**2 + (data['Longitude'] - la_coords[1])**2)
    data['dist_to_sf'] = np.sqrt((data['Latitude'] - sf_coords[0])**2 + (data['Longitude'] - sf_coords[1])**2)
    data['min_dist_to_major_city'] = np.minimum(data['dist_to_la'], data['dist_to_sf'])
    return data

if __name__ == "__main__":
    from data_loader import load_housing_data
    df = load_housing_data()
    df_feat = add_engineered_features(df)
    print("Engineered DataFrame shape:", df_feat.shape)
    print("New columns:", [c for c in df_feat.columns if c not in df.columns])
    print(df_feat[['MedInc', 'bedrooms_ratio', 'rooms_per_occupant', 'min_dist_to_major_city', 'MedHouseVal']].head())
