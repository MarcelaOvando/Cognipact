import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load data
df = pd.read_csv("master_term_parcel_matrix.csv")
cat_df = pd.read_csv("my_ratio_fileCATEGORIES_lastVP-MOT.csv", sep = ';')

# Identify columns
asso_cols = [c for c in df.columns if c.lower().startswith("asso_")]
comm_cols = [c for c in df.columns if c.lower().startswith("comm_")]

term_col = df.columns[0]
freq_col = df.columns[-1]

term_col_cat = cat_df.columns[0]
cat_col = cat_df.columns[2]

# Merge category info
df = df.merge(
    cat_df,
    left_on=term_col,
    right_on=term_col_cat,
    how="inner"
)

results = []

# Category-level analysis
for category in df[cat_col].unique():

    df_cat = df[df[cat_col] == category]

    # Sum movie frequencies
    total_freq = df_cat[freq_col].sum()
    if total_freq == 0:
        continue

    # Initialize parcel sums
    asso_sum = np.zeros(len(asso_cols))
    comm_sum = np.zeros(len(comm_cols))

    # Sum across terms (parcel-wise)
    for _, row in df_cat.iterrows():
        asso_vals = row[asso_cols].astype(float).values
        comm_vals = row[comm_cols].astype(float).values

        asso_sum += np.nan_to_num(asso_vals)
        comm_sum += np.nan_to_num(comm_vals)

    # Normalize by total frequency
    asso_norm = asso_sum / total_freq
    comm_norm = comm_sum / total_freq

    # Welch t-test
    tval, pval = ttest_ind(
        asso_norm,
        comm_norm,
        equal_var=False
    )

    results.append({
        "category": category,
        "t_value": tval,
        "p_value": pval,
        "p_mean_asso_norm": np.mean(asso_norm),
        "p_mean_comm_norm": np.mean(comm_norm),
        "n_terms": df_cat.shape[0],
        "total_movie_frequency": total_freq
    })

# Multiple comparisons
res_df = pd.DataFrame(results)
res_df["p_fdr"] = multipletests(
    res_df["p_value"],
    method="fdr_bh"
)[1]

# Save
res_df.to_csv("category_level_parcelwise_ttest_normALL-FINAL.csv", index=False)

print("Category-level parcel-wise t-tests completed.")