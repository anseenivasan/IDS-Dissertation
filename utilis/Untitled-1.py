# utils/data_loader.py
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Prerequisite: Update constants.py ---
# In constants.py, you should have:
# RAW_TO_GRANULAR_MAP = {
#     'Benign': 'Benign',
#     'BENIGN': 'Benign',
#     'DoS Hulk': 'DoS',
#     # ... all your raw to granular mappings ...
#     'Web Attack  Brute Force': 'Web Attack - Brute Force',
#     'Web Attack  Brute Force': 'Web Attack - Brute Force',
#     'Web Attack  Sql Injection': 'Web Attack - SQL Injection',
#     'Web Attack  Sql Injection': 'Web Attack - SQL Injection',
#     'Web Attack  XSS': 'Web Attack - XSS',
#     'Web Attack  XSS': 'Web Attack - XSS',
#     'SQL Injection': 'Web Attack - SQL Injection',
#     'Brute Force -Web': 'Web Attack',
#     'Brute Force -XSS': 'Web Attack',
#     'PortScan': 'PortScan',
#     'Infiltration': 'Infiltration',
#     'Infilteration': 'Infiltration',
#     'Heartbleed': 'Heartbleed',
# }
#
# BROAD_MAPPING = { # This maps from GRANULAR_HARMONIZED_LABELS (strings) to BROAD_LABELS (strings)
#     'Benign': 'Benign',
#     'DoS': 'DDoS',
#     'DDoS': 'DDoS',
#     'Botnet': 'Botnet',
#     'Infiltration': 'Infiltration',
#     'PortScan': 'PortScan',
#     'FTP-BruteForce': 'Brute Force',
#     'SSH-BruteForce': 'Brute Force',
#     'Web Attack - Brute Force': 'Web Attack',
#     'Web Attack - XSS': 'Web Attack',
#     'Web Attack - SQL Injection': 'Web Attack',
#     'Heartbleed': 'Heartbleed',
#     # Anything not explicitly mapped here will become 'Other Attack'
# }
#
# Remove 'granular_to_broad_map' from constants.py if it's redundant with BROAD_MAPPING.
# If 'granular_to_broad_map' is indeed the one you want to use for the granular-to-broad mapping,
# then rename BROAD_MAPPING to something else or remove it, and ensure create_label_mapper_function uses granular_to_broad_map.
# For this corrected code, I'll assume BROAD_MAPPING is the canonical granular-to-broad map.

from .constants import GLOBAL_FLAG_COLUMNS_WITH_NANS, BROAD_MAPPING, RAW_TO_GRANULAR_MAP # Import the new RAW_TO_GRANULAR_MAP
# from .metrics_helpers import calculate_specificity # Not directly used in data loading

# --- Helper Function to Harmonize Labels ---
def harmonize_labels(df):
    """
    Harmonizes raw labels into consistent granular labels.
    This function should ONLY produce the 'Label' column with granular harmonized strings.
    """
    df_copy = df.copy()

    # Use the map from constants.py
    df_copy['Label'] = df_copy['Label'].apply(lambda x: RAW_TO_GRANULAR_MAP.get(x, x))

    # Optional debug for granular labels
    final_granular_labels = set(df_copy['Label'].unique())
    print(f"Harmonized granular labels: {final_granular_labels}")

    return df_copy


def align_to_common_features(df, common_features):
    """
    Ensures all common_features are present in df, fills with NaN if not.
    Selects only the common features and the 'Label' column.
    """
    # Ensure 'Label' is present for alignment, even if not in common_features
    # This function should only deal with features and the primary 'Label' column.
    # 'BroadLabel' will be added later after encoding.
    cols_to_keep = list(common_features) + ['Label']
    
    for col in common_features: # Only fill NaNs for actual features
        if col not in df.columns:
            df[col] = np.nan # Add missing feature column and fill with NaN

    # Ensure 'Label' column is present before selecting
    if 'Label' not in df.columns:
        # This case should ideally not happen if load_and_preprocess_dataset is robust
        raise ValueError("DataFrame missing 'Label' column during feature alignment.")

    return df[cols_to_keep]


# --- Helper Function: create_label_mapper_function (Corrected Version) ---
def create_label_mapper_function(label_encoder, mapping_dict):
    """
    Creates a function that maps encoded granular labels to encoded broad labels.
    Returns the mapper function and the LabelEncoder for broad labels.
    """
    # 1. Decode all granular class labels that the original label_encoder knows about.
    decoded_granular_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

    # 2. Apply the broad mapping to these decoded granular labels.
    mapped_broad_labels_raw = [mapping_dict.get(lbl, 'Other Attack') for lbl in decoded_granular_labels]

    # 3. Re-introduce aggressive normalization for robustness before fitting broad_label_encoder
    mapped_broad_labels_normalized = []
    for label_str in mapped_broad_labels_raw:
        mapped_broad_labels_normalized.append(label_str.encode('utf-8').decode('utf-8').strip())
    
    # 4. Create a NEW sklearn LabelEncoder for the broad labels and fit it.
    broad_label_encoder = LabelEncoder()
    # Fit on the normalized labels to ensure consistency
    encoded_broad_labels = broad_label_encoder.fit_transform(mapped_broad_labels_normalized)

    # 5. Create a direct mapping dictionary from encoded granular integers to encoded broad integers.
    encoded_granular_to_encoded_broad_map = dict(zip(range(len(label_encoder.classes_)), encoded_broad_labels))

    # 6. Define the mapper function.
    def mapper(encoded_granular_labels_array):
        if np.isscalar(encoded_granular_labels_array):
            return encoded_granular_to_encoded_broad_map[encoded_granular_labels_array]
        return np.array([encoded_granular_to_encoded_broad_map[x] for x in encoded_granular_labels_array])

    return mapper, broad_label_encoder


# --- Helper Function to Load and Preprocess a Single Dataset ---
def load_and_preprocess_dataset(dataset_name, folder_path):
    """
    Loads Parquet files from a specified folder, preprocesses them,
    and returns a combined DataFrame along with individual processed DataFrames.
    Preprocessing includes column stripping, NaN handling in labels, label harmonization,
    and specific NaN imputation for flag columns.
    """
    print(f"\n--- Loading and Preprocessing {dataset_name} ---")
    
    if not os.path.exists(folder_path):
        print(f"Error: Dataset folder not found at {folder_path}. Please check the path.")
        return None, None

    parquet_files = [f for f in os.path.listdir(folder_path) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"No .parquet files found in {folder_path}. Please ensure data is preprocessed and saved.")
        return None, None

    processed_individual_dfs = {} # Will store fully processed individual DFs
    all_dfs_for_concat = [] # To build the combined_df

    for file_name in parquet_files:
        df_key = os.path.splitext(file_name)[0]
        file_path = os.path.join(folder_path, file_name)
        try:
            try:
                df = pd.read_parquet(file_path)
                print(f"DEBUG: Raw labels in {df_key} of {dataset_name} before harmonization: {df['Label'].dropna().unique().tolist()}")
            except Exception as read_error:
                print(f"ERROR: Failed to read Parquet file '{file_name}' for {dataset_name}: {read_error}")
                continue

            df.columns = df.columns.str.strip()

            label_series = df.get('Label')
            if label_series is None:
                print(f"ERROR: 'Label' column is entirely missing from '{df_key}' for {dataset_name}. Skipping this file.")
                continue
            elif label_series.isnull().any():
                print(f"WARNING: NaNs found in 'Label' column for '{df_key}' of {dataset_name}. Dropping affected rows.")
                df.dropna(subset=['Label'], inplace=True)

            # Robust initial cleaning of raw labels (e.g., handling '' character)
            df['Label'] = df['Label'].apply(lambda x: x.encode('utf-8', 'replace').decode('utf-8').replace('', '-').strip() if isinstance(x, str) else x)

            # Harmonize raw labels to consistent granular string labels
            df = harmonize_labels(df) # This now only produces 'Label' column with granular strings

            # Handle flag column NaNs
            initial_total_nans_individual = df.isnull().sum().sum()
            if initial_total_nans_individual > 0:
                present_flag_cols_individual = [col for col in GLOBAL_FLAG_COLUMNS_WITH_NANS if col in df.columns]
                if present_flag_cols_individual:
                    df[present_flag_cols_individual] = df[present_flag_cols_individual].fillna(0)
                
                # Drop any other remaining NaNs
                remaining_nans_individual = df.isnull().sum().sum()
                if remaining_nans_individual > 0:
                    df.dropna(inplace=True)
            
            processed_individual_dfs[df_key] = df # Store the fully processed individual DF
            all_dfs_for_concat.append(df) # Add to list for combined_df
            
        except Exception as e:
            print(f"Error processing '{file_name}' for {dataset_name} (after read): {e}")

    print(f"Total {len(processed_individual_dfs)} fully processed DataFrames for {dataset_name}.")

    combined_df = pd.concat(all_dfs_for_concat, ignore_index=True)
    print(f"Combined DataFrame shape for {dataset_name}: {combined_df.shape}")

    print(f"DEBUG: Harmonized granular string labels in {dataset_name} (before encoding): {combined_df['Label'].unique().tolist()}")
    print(f"Label distribution in {dataset_name} (after harmonization):\n{combined_df['Label'].value_counts()}")
    
    return combined_df, processed_individual_dfs


# --- Main Orchestration Function: load_and_align_all_data ---
def load_and_align_all_data(dataset_paths):
    """
    Loads and preprocesses all specified datasets, aligns their features,
    and encodes their labels consistently.
    Returns:
        tuple: (all_combined_dfs, all_individual_dfs_by_dataset, label_encoder,
                ALL_ENCODED_LABELS, common_features, broad_label_mapper, broad_label_encoder)
    """
    all_combined_dfs = {}
    all_individual_dfs_by_dataset = {}
    
    # 1. Load and preprocess each dataset individually (granular harmonization applied here)
    for name, path in dataset_paths.items():
        processed_df, individual_dfs = load_and_preprocess_dataset(name, path)
        if processed_df is not None:
            all_combined_dfs[name] = processed_df
            all_individual_dfs_by_dataset[name] = individual_dfs
    
    if not all_combined_dfs:
        print("No datasets loaded for alignment.")
        return {}, {}, None, None, [], None, None

    # 2. Align Features Across All Datasets
    print("\n--- Aligning Features Across All Datasets ---")
    first_df_key = next(iter(all_combined_dfs))
    common_features = set(all_combined_dfs[first_df_key].columns.drop('Label', errors='ignore'))

    for df_key in list(all_combined_dfs.keys())[1:]:
        df = all_combined_dfs[df_key]
        common_features &= set(df.columns.drop('Label', errors='ignore'))

    common_features = sorted(list(common_features))

    # Apply common features to all combined datasets
    for name, df in all_combined_dfs.items():
        # align_to_common_features will ensure features are aligned and 'Label' is present
        all_combined_dfs[name] = align_to_common_features(df, common_features)
        print(f"{name} shape after feature alignment (combined): {all_combined_dfs[name].shape}")
        print(f"  DEBUG: {name} Label unique values AFTER alignment (still strings): {all_combined_dfs[name]['Label'].unique().tolist()}")

    # Apply common features to all individual daily DataFrames
    for dataset_name, individual_dfs_dict in all_individual_dfs_by_dataset.items():
        for df_key, df in individual_dfs_dict.items():
            # align_to_common_features will ensure features are aligned and 'Label' is present
            individual_dfs_dict[df_key] = align_to_common_features(df, common_features)
        print(f"Individual DFs for {dataset_name} aligned to common features.")


    # 3. Encode Granular Labels Consistently
    print("\n--- Encoding Labels Consistently (Granular) ---")
    all_unique_granular_labels_raw = set()
    for df in all_combined_dfs.values():
        all_unique_granular_labels_raw.update(df['Label'].unique())
    
    # Normalize granular labels before fitting label_encoder
    normalized_granular_labels = sorted(
        list(set(label.encode('utf-8').decode('utf-8').strip() for label in all_unique_granular_labels_raw))
    )
    print(f"DEBUG: Unique granular labels collected for fitting label_encoder: {normalized_granular_labels}")

    label_encoder = LabelEncoder()
    label_encoder.fit(normalized_granular_labels)
    print(f"DEBUG: label_encoder.classes_ after fit: {label_encoder.classes_.tolist()}")

    ALL_ENCODED_LABELS = label_encoder.transform(normalized_granular_labels)
    print(f"DEBUG: ALL_ENCODED_LABELS (encoded granular labels): {ALL_ENCODED_LABELS.tolist()}")

    # Apply granular encoding to 'Label' column in all combined datasets
    for name, df in all_combined_dfs.items():
        df_labels_normalized = df['Label'].apply(lambda x: x.encode('utf-8').decode('utf-8').strip()).to_numpy()
        all_combined_dfs[name]['Label'] = label_encoder.transform(df_labels_normalized)
        print(f"DEBUG: After granular encoding for {name} (combined), first 5 encoded labels: {all_combined_dfs[name]['Label'].head().tolist()}")

    # Apply granular encoding to 'Label' column in all individual daily DataFrames
    for dataset_name, individual_dfs_dict in all_individual_dfs_by_dataset.items():
        for df_key, df in individual_dfs_dict.items():
            df_labels_normalized = df['Label'].apply(lambda x: x.encode('utf-8').decode('utf-8').strip()).to_numpy()
            individual_dfs_dict[df_key]['Label'] = label_encoder.transform(df_labels_normalized)
            print(f"DEBUG: After granular encoding for individual DF {df_key}, first 5 encoded labels: {individual_dfs_dict[df_key]['Label'].head().tolist()}")
        print(f"Labels in individual DFs for {dataset_name} encoded.")


    # 4. Create Broad Label Mapper and Encoder (from encoded granular to encoded broad)
    print("\n--- Creating Broad Label Mapper and Encoder ---")
    # This function now correctly handles normalization internally for its broad labels
    broad_label_mapper, broad_label_encoder = create_label_mapper_function(label_encoder, BROAD_MAPPING)
    print(f"Broad labels defined: {broad_label_encoder.classes_.tolist()}")

    # 5. Apply Broad Label Mapping to create 'BroadLabel' column (encoded integers)
    # This happens AFTER granular labels are encoded.
    print("\n--- Applying Broad Label Mapping to DataFrames ---")
    for name, df in all_combined_dfs.items():
        # df['Label'] now contains encoded granular integers
        all_combined_dfs[name]['BroadLabel'] = broad_label_mapper(df['Label'].to_numpy())
        print(f"DEBUG: {name} 'BroadLabel' unique values after mapping: {all_combined_dfs[name]['BroadLabel'].unique().tolist()}")

    for dataset_name, individual_dfs_dict in all_individual_dfs_by_dataset.items():
        for df_key, df in individual_dfs_dict.items():
            # df['Label'] now contains encoded granular integers
            individual_dfs_dict[df_key]['BroadLabel'] = broad_label_mapper(df['Label'].to_numpy())
            print(f"DEBUG: Individual DF {df_key} 'BroadLabel' unique values after mapping: {individual_dfs_dict[df_key]['BroadLabel'].unique().tolist()}")


    # Optional validation of mapping completeness
    all_broad_labels_in_data_encoded = set()
    for name, df in all_combined_dfs.items():
        all_broad_labels_in_data_encoded.update(df['BroadLabel'].dropna().unique())

    # Decode them for comparison with TARGET_ATTACK_LABELS_STR_BROAD (from constants)
    all_broad_labels_in_data_decoded = broad_label_encoder.inverse_transform(list(all_broad_labels_in_data_encoded))
    
    # Ensure TARGET_ATTACK_LABELS_STR_BROAD includes 'Benign' and 'Other Attack' if they are present.
    # It's good practice to ensure consistency between constants and actual data.
    from .constants import TARGET_ATTACK_LABELS_STR_BROAD # Re-import to ensure it's fresh if constants.py changed
    expected_broad_labels_str_set = set(TARGET_ATTACK_LABELS_STR_BROAD) 
    if 'Benign' not in expected_broad_labels_str_set:
        expected_broad_labels_str_set.add('Benign')
    if 'Other Attack' not in expected_broad_labels_str_set: # Add 'Other Attack' if it's a possibility
        expected_broad_labels_str_set.add('Other Attack')

    unknown_broad_labels_in_data = set(all_broad_labels_in_data_decoded) - expected_broad_labels_str_set
    if unknown_broad_labels_in_data:
        print(f"WARNING: Found BroadLabels (decoded) in data not in TARGET_ATTACK_LABELS_STR_BROAD: {unknown_broad_labels_in_data}")
    else:
        print("All BroadLabels (decoded) match expected broad categories.")

    missing_expected_broad_labels = expected_broad_labels_str_set - set(all_broad_labels_in_data_decoded)
    if missing_expected_broad_labels:
        print(f"INFO: Expected BroadLabels (from TARGET_ATTACK_LABELS_STR_BROAD) not found in data: {missing_expected_broad_labels}")


    return (
            all_combined_dfs,
            all_individual_dfs_by_dataset,
            label_encoder,
            ALL_ENCODED_LABELS, # These are granular encoded labels
            list(common_features),
            broad_label_mapper,
            broad_label_encoder
        )