import os
import pandas as pd

# Define file paths
file_stable_09 = "data/STable_09_Sel_STARSOutput.xlsx"
file_stable_15 = "data/STable_15_IFNg_data_STARS.xlsx"
output_file = "output/gRNA_TargetDNA_AverageScore_Combined.xlsx"

def process_data():
    try:
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        # Load STable_09 with the first row skipped
        df_stable_09 = pd.read_excel(file_stable_09, skiprows=1)
        df_stable_15 = pd.read_excel(file_stable_15)

        # Print column names for verification (optional)
        print("Columns in STable_09 (after skipping rows):", df_stable_09.columns)
        print("Columns in STable_15:", df_stable_15.columns)

        # Verify columns for STable_09
        if "Perturbations" not in df_stable_09.columns or "Average Score" not in df_stable_09.columns:
            raise KeyError("Required columns not found in STable_09 after skipping rows")

        # Verify columns for STable_15
        if "sgRNA" not in df_stable_15.columns or "Average log fold change IFNgamma - mock" not in df_stable_15.columns:
            raise KeyError("Required columns not found in STable_15")

        # Process STable_09
        df_stable_09_extracted = df_stable_09[["Perturbations", "Average Score"]].copy()
        df_stable_09_extracted["gRNA"] = df_stable_09["Perturbations"].str.split(";").str[0]
        df_stable_09_extracted.rename(columns={"Perturbations": "Target DNA"}, inplace=True)

        # Process STable_15
        df_stable_15_extracted = df_stable_15[["sgRNA", "Average log fold change IFNgamma - mock"]].copy()
        df_stable_15_extracted.rename(columns={"sgRNA": "gRNA", "Average log fold change IFNgamma - mock": "Average Score"}, inplace=True)
        df_stable_15_extracted["Target DNA"] = None

        # Combine both datasets
        combined_df = pd.concat([df_stable_09_extracted, df_stable_15_extracted], ignore_index=True)

        # Save the combined data
        combined_df.to_excel(output_file, index=False)
        print(f"Combined data has been saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_data()
