import pickle
from sklearn.preprocessing import OrdinalEncoder

# Load the OrdinalEncoder from the .pkl file
with open('location_ordinal_encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)

# Check if it's truly an OrdinalEncoder
if isinstance(ordinal_encoder, OrdinalEncoder):
    # Initialize a dictionary to store mappings for each feature
    encoding_mappings = {}
    
    # Loop through each feature's categories
    for feature_idx, categories in enumerate(ordinal_encoder.categories_):
        # Create a dictionary: {category: encoded_value}
        feature_mapping = {category: idx for idx, category in enumerate(categories)}
        encoding_mappings[f"Feature_{feature_idx}"] = feature_mapping
    
    # Print the full mapping
    print("OrdinalEncoder Mappings:")
    print(encoding_mappings)
else:
    print("The loaded object is not an OrdinalEncoder.")