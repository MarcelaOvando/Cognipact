"""
Cognitive term activation analysis: Normalised difference in activation
frequency between association and commissural parcels.
Computes: (association - commissural) / (association + commissural)
Used for: Figure 4 barplot values
"""
import pandas as pd
import numpy as np

# Step 1: Load the data from the CSV files
asso_counts_file = 'term_counts_by_parcel_asso2.csv'
commissural_counts_file = 'term_counts_by_parcel_comm2.csv'
reference_counts_file = 'reference_term_counts.csv'

# Load data
asso_df = pd.read_csv(asso_counts_file, index_col="Term")
comm_df = pd.read_csv(commissural_counts_file, index_col="Term")
ref_df = pd.read_csv(reference_counts_file)

# Convert Reference_Counts to a dictionary for quick lookup
ref_counts = dict(zip(ref_df["Term"], ref_df["Reference_Count"]))

# All terms
all_terms = sorted(set(asso_df.index).union(set(comm_df.index)))

# Step 2: Prepare a DataFrame with normalized values and calculated differences
data = []
for term in all_terms:
    ref = ref_counts.get(term, 1)  # Avoid division by zero
    # Normalize term activations by reference count
    asso = asso_df.loc[term].sum() / ref if term in asso_df.index else 0
    comm = comm_df.loc[term].sum() / ref if term in comm_df.index else 0
    # Normalized Difference calculation
    if asso + comm != 0:
        normalized_diff = (asso - comm) / (asso + comm)
    else:
        normalized_diff = 0
    data.append((term, asso, comm, normalized_diff))

# Create DataFrame
df = pd.DataFrame(data, columns=["Term", "Asso_Normalized",
                                   "Commissural_Normalized",
                                   "Normalized_Difference"])

# Step 3: Save results
df.to_csv('normalized_difference_results.csv', index=False)

print("Final Results (All terms):")
print(df)