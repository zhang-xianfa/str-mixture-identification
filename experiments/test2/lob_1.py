import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
# 请替换为您的实际文件路径
processed_file_path = r"..\..\data\processed_data1.xlsx"

# --- 1. Load Preprocessed Data ---
print(f"Loading preprocessed data from: {processed_file_path}")
try:
    df_processed = pd.read_excel(processed_file_path)
    print("Data loaded successfully.")
    print(f"Shape of processed data: {df_processed.shape}")
    # print("Processed data columns:", df_processed.columns.tolist())
    # print("Processed data head:")
    # print(df_processed.head())

except FileNotFoundError:
    print(f"Error: Processed file not found, please check the path: {processed_file_path}")
    exit()
except Exception as e:
    print(f"Error loading processed file: {e}")
    exit()

# Ensure essential columns are present and correct types
required_cols = ['Sample File', 'Marker', 'Height', 'Size', 'Allele', 'NOC', 'Is_OL', 'Is_Stutter_Candidate', 'Is_PlusA_Candidate']
if not all(col in df_processed.columns for col in required_cols):
     missing = [col for col in required_cols if col not in df_processed.columns]
     print(f"Error: Processed data is missing required columns: {missing}")
     exit()

# Convert relevant columns to numeric, coercing errors
df_processed['Height'] = pd.to_numeric(df_processed['Height'], errors='coerce')
df_processed['Size'] = pd.to_numeric(df_processed['Size'], errors='coerce')
# Allele might contain non-numeric values like 'OL', keep as string for filtering/flags
df_processed['Allele'] = df_processed['Allele'].astype(str)
df_processed['NOC'] = pd.to_numeric(df_processed['NOC'], errors='coerce').astype('Int64') # Use Int64 to allow NaN integers if necessary

# Drop rows with missing essential data points (should be minimal after preproc, but safe)
initial_rows_count = df_processed.shape[0]
df_processed.dropna(subset=['NOC', 'Height', 'Size'], inplace=True)
rows_after_dropna = df_processed.shape[0]
if initial_rows_count > rows_after_dropna:
     print(f"Warning: Removed {initial_rows_count - rows_after_dropna} rows with missing essential data (NOC, Height, or Size).")

# --- 2. Feature Engineering (Aggregate per Sample File) ---
# Now we process the long format data to create features at the sample level.
print("\nPerforming feature engineering per Sample File...")

sample_features = {}

# Feature 1: Total number of peaks per sample
# Explicitly name the Series
total_peak_count = df_processed.groupby('Sample File').size()
total_peak_count.name = 'total_peak_count' # <--- Set the name here
sample_features['total_peak_count'] = total_peak_count


# Feature 2: Number of peaks per marker per sample (pivot to wide format)
# This is a powerful feature, indicating allele counts per locus
marker_peak_counts = df_processed.groupby(['Sample File', 'Marker']).size().unstack(fill_value=0)
# Rename columns for clarity (e.g., D8S1179 -> Marker_D8S1179_peak_count)
marker_peak_counts.columns = [f'Marker_{col}_peak_count' for col in marker_peak_counts.columns]
sample_features['marker_peak_counts'] = marker_peak_counts

# Feature 3: Presence/Count of artifact peaks per sample
# artifact_flags is already a DataFrame from groupby.any()
artifact_flags = df_processed.groupby('Sample File')[['Is_OL', 'Is_Stutter_Candidate', 'Is_PlusA_Candidate']].any().astype(int)
# artifact_counts is already a DataFrame from groupby.sum()
artifact_counts = df_processed.groupby('Sample File')[['Is_OL', 'Is_Stutter_Candidate', 'Is_PlusA_Candidate']].sum()
artifact_counts.columns = ['total_OL_peaks', 'total_Stutter_candidates', 'total_PlusA_candidates'] # Use consistent naming

sample_features['has_OL_peak'] = artifact_flags['Is_OL']
sample_features['has_Stutter_candidate'] = artifact_flags['Is_Stutter_Candidate']
sample_features['has_PlusA_candidate'] = artifact_flags['Is_PlusA_Candidate']
sample_features['total_OL_peaks'] = artifact_counts['total_OL_peaks']
sample_features['total_Stutter_candidates'] = artifact_counts['total_Stutter_candidates']
sample_features['total_PlusA_candidates'] = artifact_counts['total_PlusA_candidates']


# Feature 4: Basic stats on Height and Size across all peaks in the sample
numeric_cols_for_stats = df_processed.dropna(subset=['Height', 'Size']) # Use the dataframe already cleaned for essential NaNs
if not numeric_cols_for_stats.empty:
    # Using .agg() with a dictionary to name output columns directly
    height_stats = numeric_cols_for_stats.groupby('Sample File')['Height'].agg(
        Avg_Height='mean',
        Median_Height='median',
        Std_Height='std',
        Max_Height='max',
        Min_Height='min',
        Skew_Height=lambda x: x.skew() if len(x) > 2 else 0, # Skew needs at least 3 points
        Kurtosis_Height=lambda x: x.kurtosis() if len(x) > 3 else 0 # Kurtosis needs at least 4 points
    )
    # Handle cases where std/skew/kurtosis are NaN/Inf due to few data points
    height_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    height_stats.fillna(0, inplace=True) # Fill NaN stats with 0

    size_stats = numeric_cols_for_stats.groupby('Sample File')['Size'].agg(
        Avg_Size='mean',
        Median_Size='median',
        Std_Size='std',
        Max_Size='max',
        Min_Size='min'
    )
    size_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    size_stats.fillna(0, inplace=True) # Fill NaN stats with 0


    sample_features['height_stats'] = height_stats
    sample_features['size_stats'] = size_stats
else:
    print("Warning: No numeric peak data available for height/size stats calculation.")
    # Create empty dataframes to avoid errors later
    sample_features['height_stats'] = pd.DataFrame(index=df_processed['Sample File'].unique(), columns=['Avg_Height', 'Median_Height', 'Std_Height', 'Max_Height', 'Min_Height', 'Skew_Height', 'Kurtosis_Height']).fillna(0)
    sample_features['size_stats'] = pd.DataFrame(index=df_processed['Sample File'].unique(), columns=['Avg_Size', 'Median_Size', 'Std_Size', 'Max_Size', 'Min_Size']).fillna(0)


# Feature 5: AMEL specific features
# Check presence of X and Y peaks specifically
amel_alleles = df_processed[df_processed['Marker'] == 'AMEL'].copy()

# Calculate presence flags - results in Series with Sample File index
# Explicitly name the Series here
has_X = (amel_alleles[amel_alleles['Allele'] == 'X'].groupby('Sample File').size() > 0).astype(int)
has_Y = (amel_alleles[amel_alleles['Allele'] == 'Y'].groupby('Sample File').size() > 0).astype(int)

# Reindex to ensure all samples are included, fill missing with 0 (meaning X or Y is not present)
all_samples = df_processed['Sample File'].unique()
has_X = has_X.reindex(all_samples, fill_value=0)
has_Y = has_Y.reindex(all_samples, fill_value=0)

# Set the name of the Series - <--- Set the name here
has_X.name = 'has_X_AMEL'
has_Y.name = 'has_Y_AMEL'

sample_features['has_X_AMEL'] = has_X
sample_features['has_Y_AMEL'] = has_Y


# Combine all features into a single DataFrame X
# Start with a dataframe that has the correct index (all sample files)
X = pd.DataFrame(index=all_samples)

# Merge features - now the Series/DataFrames should have names or are DataFrames already
X = X.merge(sample_features['total_peak_count'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['marker_peak_counts'], left_index=True, right_index=True, how='left')

if 'height_stats' in sample_features and not sample_features['height_stats'].empty:
    X = X.merge(sample_features['height_stats'], left_index=True, right_index=True, how='left')

if 'size_stats' in sample_features and not sample_features['size_stats'].empty:
     X = X.merge(sample_features['size_stats'], left_index=True, right_index=True, how='left')

# These are Series with names now
X = X.merge(sample_features['has_OL_peak'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['has_Stutter_candidate'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['has_PlusA_candidate'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['total_OL_peaks'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['total_Stutter_candidates'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['total_PlusA_candidates'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['has_X_AMEL'], left_index=True, right_index=True, how='left')
X = X.merge(sample_features['has_Y_AMEL'], left_index=True, right_index=True, how='left')


# Ensure all columns have numeric values (fill NaNs created by merge/unstack for samples missing certain markers)
X.fillna(0, inplace=True)

# Define target variable y (NOC) - needs to be aligned with X's index (Sample File)
# Use .first() as NOC should be the same for all rows of a sample
y = df_processed.groupby('Sample File')['NOC'].first()

# Align X and y indices - Use inner join to ensure only samples present in both X and y are kept
X, y = X.align(y, join='inner', axis=0)

print(f"\nShape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}")
# print("Features head:")
# print(X.head())
# print("Target head:")
# print(y.head())

# --- 3. Prepare data for XGBoost ---
# Convert NOC labels to integers for classification
# Note: NOC is already Int64, but XGBoost multi:softmax expects integer labels 0 to num_class-1
# Check unique values of y
unique_y_values = y.unique()
print(f"\nUnique NOC values in target variable: {sorted(unique_y_values.tolist())}")

# Let's shift them to start from 0 if necessary for the objective
# XGBoost multi:softmax expects integer labels 0 to num_class-1
# Check if min value is > 0. If so, subtract min_y.
min_y = y.min()
if min_y > 0:
    y_encoded = y - min_y
    print(f"Shifting target variable (NOC) by {min_y} to start from 0.")
else:
    y_encoded = y
    print("Target variable (NOC) already starts from 0 or less.")

y_encoded = y_encoded.astype(int)
num_classes = len(y_encoded.unique())
print(f"Number of classes for prediction: {num_classes}")

# Check if there's enough data for splitting after processing and alignment
if X.shape[0] < 2 or num_classes < 2:
    print("\nNot enough data or classes to perform train/test split and train a classifier.")
    print(f"Number of samples: {X.shape[0]}, Number of classes: {num_classes}")
    exit()


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 4. Train and Evaluate XGBoost Model ---
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes, # Use the calculated number of classes
    n_estimators=800,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    max_bin=1024,
    reg_alpha=0.2,
    reg_lambda=0.2,
    tree_method='hist',
    # enable_categorical=True, # This is for native categorical feature support, not for encoded features
    eval_metric='mlogloss',
    use_label_encoder=False, # Deprecated, should be False
    random_state=42
)

# If you have actual categorical features that were label encoded or one-hot encoded,
# enable_categorical=True should be removed and standard encoding used.
# Here, 'Marker' names were one-hot encoded by unstacking, which is fine.
# If other categorical features were present (like 'Dye'), they would need encoding.
# Let's assume the current features are all numeric or binary flags.

# Fit the model
model.fit(X_train, y_train)
print("Model training complete.")

# Predict and evaluate
print("\nEvaluating model on test set...")
y_pred = model.predict(X_test)

# Shift predictions back if target variable was shifted
y_test_original = y_test + min_y
y_pred_original = y_pred + min_y

print("\nClassification Report:")
# Use target_names for better report labels if original NOC values are desired
# Create target_names from unique original y values
target_names = [str(val) for val in sorted(y.unique().tolist())]
print(classification_report(y_test_original, y_pred_original, digits=4, target_names=target_names))

# --- 5. Feature Importance ---
print("\nTop Features:")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.head(20)) # Print top 20 features
