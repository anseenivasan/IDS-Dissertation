# utils/constants.py
GLOBAL_FLAG_COLUMNS_WITH_NANS = ['Fwd URG Flags', 'CWE Flag Count']

DATASET_PATHS = {
    'CIC_IDS_2018': '/Users/srinara2/Downloads/Dissertation_research/cic-ids2018-ml/parquet_output',
    'CIC_IDS_2017': '/Users/srinara2/Downloads/Dissertation_research/cic-ids2017-ml/parquet_output'
}

# Mapping from raw dataset labels to consistently granular harmonized labels (strings)
RAW_TO_GRANULAR_MAP = {
    # Benign
    'Benign': 'Benign',
    'BENIGN': 'Benign',

    # DoS
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS attacks-Hulk': 'DoS',
    'DoS attacks-GoldenEye': 'DoS',
    'DoS attacks-Slowloris': 'DoS',
    'DoS attacks-SlowHTTPTest': 'DoS',

    # DDoS
    'DDoS': 'DDoS',
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDOS attack-HOIC': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',

    # Botnet
    'Bot': 'Botnet',

    # Brute Force
    'FTP-Patator': 'FTP-BruteForce',
    'SSH-Patator': 'SSH-BruteForce',
    'FTP-BruteForce': 'FTP-BruteForce',
    'SSH-BruteForce': 'SSH-BruteForce',
    'SSH-Bruteforce': 'SSH-BruteForce', # Added for consistency

    # Web Attacks (Note: 'Web Attack' is a broader category, but these are granular results)
    'Web Attack  Brute Force': 'Web Attack - Brute Force',
    'Web Attack  Brute Force': 'Web Attack - Brute Force',
    'Web Attack  Sql Injection': 'Web Attack - SQL Injection',
    'Web Attack  Sql Injection': 'Web Attack - SQL Injection',
    'Web Attack � Sql Injection': 'Web Attack - SQL Injection',
    'Web Attack  XSS': 'Web Attack - XSS',
    'Web Attack  XSS': 'Web Attack - XSS',
    'Web Attack � XSS': 'Web Attack - XSS',
    'SQL Injection': 'Web Attack - SQL Injection', # This maps a raw label to a granular one
    'Brute Force -Web': 'Web Attack - Brute Force', # Mapping raw to specific granular
    'Brute Force -XSS': 'Web Attack - XSS', # Mapping raw to specific granular
    'Web Attack � Brute Force': 'Web Attack - Brute Force',

    # Others
    'PortScan': 'PortScan',
    'Infiltration': 'Infiltration',
    'Infilteration': 'Infiltration', # Harmonize spelling
    'Heartbleed': 'Heartbleed',
}


# Harmonized labels at the most granular level (output of harmonize_labels function)
# These are the labels you expect after the initial harmonization.
GRANULAR_HARMONIZED_LABELS = [
    'Benign', 'DoS', 'DDoS', 'Botnet', 'Infiltration', 'PortScan',
    'FTP-BruteForce', 'SSH-BruteForce',
    'Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - SQL Injection',
    'Heartbleed'
]


# Mapping from granular harmonized labels (strings) to broader categories (strings)
# This is the canonical map for granular-to-broad conversion.
BROAD_MAPPING = {
    'Benign': 'Benign',
    'DoS': 'DoS', # Group DoS into DDoS
    'DDoS': 'DDoS',
    'Botnet': 'Botnet',
    'Infiltration': 'Infiltration',
    'PortScan': 'PortScan',
    'FTP-BruteForce': 'Brute Force', # Group all Brute Force
    'SSH-BruteForce': 'Brute Force',
    'Web Attack - Brute Force': 'Web Attack - Brute Force', # Group all Web Attacks
    'Web Attack - XSS': 'Web Attack - XSS',
    'Web Attack - SQL Injection': 'Web Attack - SQL Injection',
    'Heartbleed': 'Heartbleed',
    # Any granular label not in this map will become 'Other Attack'
}

# Target labels for specific reporting (should match the values in BROAD_MAPPING)
TARGET_ATTACK_LABELS_STR_BROAD = [
    'DDoS', 'Botnet', 'Infiltration', 'PortScan',
    'Brute Force', 'Web Attack', 'Heartbleed', 'Other Attack',
     'Web Attack - Brute Force', # <-- ADDED: Now a distinct broad category
    'Web Attack - XSS', # <-- ADDED: Now a distinct broad category
    'Web Attack - SQL Injection', # <-- ADDED: Now a distinct broad categor # Include 'Other Attack' if it's a possibility
]


# For intra-dataset, you might want to use the granular labels
TARGET_ATTACK_LABELS_STR_GRANULAR = GRANULAR_HARMONIZED_LABELS # Or a subset of them


# Hyperparameter Grids (can also be here or in main evaluation files)
PARAM_GRIDS = {
    # In utilis/constants.py
    'Logistic Regression': {
        'C': [0.01, 1], # Fewer C values
        'penalty': ['l2'], # Focus on l2 first, or just one if time is critical
        'solver': ['saga'],
        'max_iter': [5000, 10000], # Significantly reduced max_iter
        'tol': [ 0.005] # Increased tolerance
    },
    'Random Forest': {
        'n_estimators': [80], # Reduced search space, consider even smaller like [20, 50] for HPO
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'max_depth': [20, 30, 40] # Add max_depth to control tree depth
    },
    'XGBoost': {
        'n_estimators': [100,200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
}