# main_evaluation.py
from  utilis.Data_loader import load_and_align_all_data
# Removed granular_to_broad_map import as it's no longer used directly here

from utilis.constants import DATASET_PATHS, TARGET_ATTACK_LABELS_STR_BROAD,PARAM_GRIDS

from cross_evaluation import run_cross_dataset_evaluation
from intra_evaluation import run_intra_dataset_evaluation

def main():
    """
    Main function to orchestrate data loading, preprocessing,
    and cross-dataset/intra-dataset evaluation.
    """
    print("--- Starting Main Evaluation Script ---")

    
    # --- SCENARIO CONFIGURATION ---
    RUN_FOCUSED_SCENARIO = True # Set to True to focus on specific labels, False for full evaluation

    if RUN_FOCUSED_SCENARIO:
        print("\n--- Running Focused Evaluation: Benign, DoS, DDoS, Botnet, Infiltration ---")
        # Define the broad string labels to keep for this focused scenario
        focused_broad_labels_to_keep = ['Benign', 'DoS', 'DDoS', 'Botnet', 'Infiltration']
        
        # Define the specific attack labels to report for this focused scenario
        # This list should only contain the attack classes you want to see in the specific metrics report.
        # 'Benign' is usually handled separately.
        focused_target_report_labels = ['DoS', 'DDoS', 'Botnet', 'Infiltration']

        all_combined_dfs, all_individual_dfs_by_dataset, label_encoder, \
        ALL_ENCODED_LABELS, common_features, broad_label_mapper, broad_label_encoder = \
            load_and_align_all_data(DATASET_PATHS, focus_labels=focused_broad_labels_to_keep)
        
    
        
        
        # Use the focused report labels for the evaluation functions
        current_target_report_labels = focused_target_report_labels
    else:
        print("\n--- Running Full Multi-Class Evaluation ---")
        all_combined_dfs, all_individual_dfs_by_dataset, label_encoder, \
        ALL_ENCODED_LABELS, common_features, broad_label_mapper, broad_label_encoder = \
            load_and_align_all_data(DATASET_PATHS)
    

        
        # Use the default target report labels from constants
        current_target_report_labels = TARGET_ATTACK_LABELS_STR_BROAD
    # --- END SCENARIO CONFIGURATION ---


    # --- ADD THIS VERIFICATION CODE HERE ---
    print("\n--- Verifying 'Infiltration' distribution across datasets ---")
    
    if 'CIC_IDS_2018' in all_combined_dfs:
        print("\n--- Broad Label Distribution in CIC_IDS_2018 ---")
        cic2018_broad_counts = all_combined_dfs['CIC_IDS_2018']['BroadLabel'].value_counts()
        decoded_2018_counts = {broad_label_encoder.inverse_transform([idx])[0]: count for idx, count in cic2018_broad_counts.items()}
        print(decoded_2018_counts)
    else:
        print("CIC_IDS_2018 not loaded or filtered out.")

    if 'CIC_IDS_2017' in all_combined_dfs:
        print("\n--- Broad Label Distribution in CIC_IDS_2017 ---")
        cic2017_broad_counts = all_combined_dfs['CIC_IDS_2017']['BroadLabel'].value_counts()
        decoded_2017_counts = {broad_label_encoder.inverse_transform([idx])[0]: count for idx, count in cic2017_broad_counts.items()}
        print(decoded_2017_counts)
    else:
        print("CIC_IDS_2017 not loaded or filtered out.")
    print("-----------------------------------------------------------\n")
    # --- END VERIFICATION CODE ---
    
    # --- Harmonized Label Distribution Check ---
    # This check is performed on the data after granular harmonization and encoding.
    # The labels are decoded back to strings for display purposes.
    print("\n--- Harmonized Label Distribution Check ---")
    
    for dataset_name, df in all_combined_dfs.items():
        print(f"\nLabel distribution for {dataset_name}:")
        
        # Granular Label Distribution (decoded for readability)
        granular_counts_encoded = df['Label'].value_counts()
        granular_labels_decoded = label_encoder.inverse_transform(granular_counts_encoded.index)
        print("Granular Labels (decoded):")
        for i, count in enumerate(granular_counts_encoded):
            print(f"  {granular_labels_decoded[i]}: {count}")

        # Broad Label Distribution (decoded for readability)
        # 'BroadLabel' column is now correctly created and encoded in load_and_align_all_data
        broad_counts_encoded = df['BroadLabel'].value_counts()
        broad_labels_decoded = broad_label_encoder.inverse_transform(broad_counts_encoded.index)
        print("\nBroad Labels (decoded):")
        for i, count in enumerate(broad_counts_encoded):
            print(f"  {broad_labels_decoded[i]}: {count}")

        # Evaluate broad label coverage against TARGET_ATTACK_LABELS_STR_BROAD
        broad_labels_in_df_decoded = set(broad_labels_decoded)
        
        # Ensure TARGET_ATTACK_LABELS_STR_BROAD includes 'Benign' and 'Other Attack' for comparison
        expected_labels_set = set(current_target_report_labels)
        if 'Benign' not in expected_labels_set:
            expected_labels_set.add('Benign')
        if 'Other Attack' not in expected_labels_set:
            expected_labels_set.add('Other Attack')

        unaccounted_labels = broad_labels_in_df_decoded - expected_labels_set
        if unaccounted_labels:
            print(f"  WARNING: Broad labels in {dataset_name} not covered by TARGET_ATTACK_LABELS_STR_BROAD + 'Benign'/'Other Attack': {unaccounted_labels}")

        missing_labels = expected_labels_set - broad_labels_in_df_decoded
        if missing_labels:
            print(f"  INFO: TARGET_ATTACK_LABELS_STR_BROAD includes broad labels not found in {dataset_name}: {missing_labels}")
    print("-------------------------------------------\n")


    # Run Cross-Dataset Evaluation
    if len(all_combined_dfs) >= 2:
        cross_results = run_cross_dataset_evaluation(
            all_combined_dfs,
            all_individual_dfs_by_dataset,
            label_encoder,          # Granular LabelEncoder
            ALL_ENCODED_LABELS,     # All encoded granular labels
            common_features,
            broad_label_mapper,     # Function to map granular encoded to broad encoded
            broad_label_encoder,
            current_target_report_labels # LabelEncoder for broad labels
        )
        # You can then print/save cross_results
    else:
        print("Skipping Cross-Dataset Evaluation: Less than two datasets loaded.")

    # Run Intra-Dataset Evaluation
    # Corrected: Pass all required arguments including broad_label_mapper and broad_label_encoder
    intra_results = run_intra_dataset_evaluation(
        all_combined_dfs,
        label_encoder,
        ALL_ENCODED_LABELS,
        common_features,
        broad_label_mapper,     # Function to map granular encoded to broad encoded
        broad_label_encoder,     # LabelEncoder for broad labels
        current_target_report_labels # <-- Pass the current target labels
    )
    # You can then print/save intra_results

if __name__ == "__main__":
    main()