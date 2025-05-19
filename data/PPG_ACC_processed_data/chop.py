import pandas as pd
import os

# Define file paths
input_file = 'data/PPG_ACC_processed_data/data.csv'
# Changed output file name to reflect the new logic
output_file = 'data/PPG_ACC_processed_data/data_short_diverse_activity.csv'
output_dir = os.path.dirname(output_file)

# --- IMPORTANT: Specify the correct column name for 'activity' ---
# Replace 'activity_label' with the actual name of the column in your CSV
# that identifies the different activities.
activity_column_name = 'activity'  # <<<--- USER: VERIFY AND UPDATE THIS VALUE

# Create output directory if it doesn't exist
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Read the CSV file
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Warning: Input file '{input_file}' is empty. Output will be an empty CSV.")
    # Attempt to read header for column names if file is header-only
    try:
        header_df = pd.read_csv(input_file, nrows=0)
        df = pd.DataFrame(columns=header_df.columns)
    except Exception: # If truly empty or header read fails
        df = pd.DataFrame()
except Exception as e:
    print(f"Error reading CSV file '{input_file}': {e}")
    exit()

# Handle case where DataFrame is empty after reading
if df.empty:
    print(f"Input DataFrame from '{input_file}' is empty. Creating an empty output file: '{output_file}'.")
    df.to_csv(output_file, index=False) # df might have columns or be completely empty
    print(f"Successfully created empty '{output_file}'.")
    exit()

# Check if the specified activity column exists in the DataFrame
if activity_column_name not in df.columns:
    print(f"Error: Activity column '{activity_column_name}' not found in the CSV.")
    print(f"Available columns are: {list(df.columns)}")
    print(f"Please update the 'activity_column_name' variable in the script with a valid column name.")
    exit()

# Group by the activity column and select the first 1/5th of rows from each group
try:
    # group_keys=False prevents the group names from becoming an index level in the result.
    # .head(len(group) // 5) takes the first 1/5th of rows for each group.
    # .reset_index(drop=True) creates a new default integer index for the final DataFrame.
    df_shortened = df.groupby(activity_column_name, group_keys=False).apply(
        lambda x: x.head(len(x) // 5)
    ).reset_index(drop=True)
except Exception as e:
    print(f"Error during grouping and processing data: {e}")
    print("An empty output file will be created.")
    df_shortened = pd.DataFrame(columns=df.columns) # Create empty DataFrame with original columns

# Write the shortened dataframe to a new CSV file
try:
    df_shortened.to_csv(output_file, index=False)
    n_rows_shortened = len(df_shortened)
    n_rows_original = len(df)
    print(f"Successfully created '{output_file}' with {n_rows_shortened} rows.")
    print(f"Original dataframe had {n_rows_original} rows.")
    print(f"Selected approximately 1/5th of rows for each group in the '{activity_column_name}' column.")
    if n_rows_shortened == 0 and n_rows_original > 0:
        print(f"Warning: The resulting dataframe is empty. This might be because all activity groups had fewer than 5 rows, or the '{activity_column_name}' column has issues (e.g., all NaN values).")

except Exception as e:
    print(f"Error writing to output file '{output_file}': {e}")
    exit()