## after RFHmCCA tree train and membership
# find Homogeneous leaf node as subgroup candidate (for many trees)
# from analysis_tree import tree, membership
from RFHmCCA import TrainTestCcaPvalue, sig_num_pvalue
import numpy as np

def compute_mcca(x_subset, y_subset, z_subset):
    mcca_pvalue = TrainTestCcaPvalue(x_subset, y_subset, z_subset, permute=1000, nums=10, train_ratio=0.5)
    mcca_sig_p = sig_num_pvalue(mcca_pvalue)
    return mcca_sig_p

def HomoSubgroupDetect(x, y, z, membership):
    unique_values = membership['node'].unique()
    # loop every leaf node
    results_leafHomo = []
    for leafi in range(len(unique_values)):
        subset_indices = np.where(membership == unique_values[leafi])[0]
        x_subset = x[subset_indices, :]
        y_subset = y[subset_indices, :]
        z_subset = z[subset_indices, :]

        HomeIndex = compute_mcca(x_subset, y_subset, z_subset)

        print(f"mCCA result for subgroup {unique_values[leafi]}: {HomeIndex}")

        result_entry = {
            'leafIndex': unique_values[leafi],
            'SubsetIndices': subset_indices.tolist(),
            'HomeIndex': HomeIndex.tolist()
        }
        results_leafHomo.append(result_entry) # list,length = num of leaf node;
    return results_leafHomo

## find leaf node with H =10;
def HomoSubgroupGet(tree, membership):
    # get variable saved in tree
    x= tree.x
    y= tree.y
    z= tree.z

    leafHomo = HomoSubgroupDetect(x, y, z, membership)
    leafIndex = []
    subsetIndices = []
    for entry in leafHomo:
        if entry['HomeIndex'] >= 1:
            leafIndex.append(entry['leafIndex']) # which leaf node in given tree;
            subsetIndices.append(entry['SubsetIndices']) # subjects ID index in this Homo =10 leaf node;
            print(f"leafIndex: {leafIndex}")
            print(f"Subset Indices: {subsetIndices}")
    return leafIndex, subsetIndices









