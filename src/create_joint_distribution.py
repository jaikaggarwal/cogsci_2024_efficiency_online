
from utils.core_utils import *
from utils.data_utils import load_intensifiers
from scipy.stats import binned_statistic_dd
tqdm.pandas()


def get_bin_edges(data, num_bins, columns_of_interest):
    bin_edges = []
    for feature in columns_of_interest:
        bin_edges.append(np.quantile(data[feature], np.linspace(0, 1, num_bins + 1)))
    bin_edges = np.array(bin_edges)
    bin_edges[:, 0] = 0
    bin_edges[:, -1] = 1
    return bin_edges


def get_bins(data, bin_edges, columns_of_interest):
    """Return the bin index that each data point in data falls into, given the space
    is subdivided to have num_bins equally sized bins.

    A bin number of i means that the corresponding value is between bin_edges[i-1], bin_edges[i]

    Returns both the bin index as a unique integer, as well as in terms of a 5d
    array corresponding to each dimension.
    """
    # Initialize uniformly-sized bins


    data = data.to_numpy()
    
    stats, edges, unraveled_binnumber = binned_statistic_dd(data, np.arange(len(data)),
                                                            statistic="mean",
                                                            bins=bin_edges,
                                                            expand_binnumbers=True)

    # Return the bin IDs
    return unraveled_binnumber.transpose()


def get_bin_names(arr, namespace):
    """
    Convert a bin's score on each dimension to the full bin name (along the lines of V1A1D2P3F4).

    Args:
        arr (np.array): scores of a given bin on each for each of the features
        namespace (str): string representation of the features we care about (e.g. VADPF)
    """
    features = np.array(list(namespace))
    added = np.char.add(features, arr.astype(str))
    names = np.sum(added.astype(object), axis=1)
    return names



def load_2019_bin_edges(num_quantiles, markers_of_interest, columns_of_interest):
    """
    Create the VADPF features scores for each sentence in our dataset, including only the markers of interest.
    """
    # ten_k_sample = Serialization.load_obj("luo_data_2019_10k_sample")
    print("Loading 2019 bin edges")
    data_dir = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/full_features/"
    files = sorted(os.listdir(data_dir))
    print(len(files))
    print(files)
    dfs = []
    for file in tqdm(files):
        df = pd.read_csv(data_dir + file)
        # df = df[df['id'].isin(ten_k_sample['id'])]
        dfs.append(df)
    df = pd.concat(dfs)
    del dfs
    df = df.set_index("id")
    df['rel_marker'] = df['rel_marker'].progress_apply(lambda x: x.strip("['']"))
    df = df[df['rel_marker'].isin(markers_of_interest)]

    print(df.shape)
    # if df.shape[0] >= 6000000:
    #     return
    print("Rescaling Politeness")
    df['Politeness'] = (df["Politeness"] - df['Politeness'].min())/(df['Politeness'].max() - df['Politeness'].min())
    return get_bin_edges(df[columns_of_interest], num_quantiles, columns_of_interest)



def load_data_from_raw(data_dir, num_quantiles, save_suffix, markers_of_interest, namespace, columns_of_interest):
    """
    Create the VADPF features scores for each sentence in our dataset, including only the markers of interest.
    """
    # ten_k_sample = Serialization.load_obj("luo_data_2019_10k_sample")
    files = sorted(os.listdir(data_dir))
    print(len(files))
    print(files)
    dfs = []
    for file in tqdm(files):
        df = pd.read_csv(data_dir + file)
        # df = df[df['id'].isin(ten_k_sample['id'])]
        dfs.append(df)
    df = pd.concat(dfs)
    del dfs
    df = df.set_index("id")
    df['rel_marker'] = df['rel_marker'].progress_apply(lambda x: x.strip("['']"))
    df = df[df['rel_marker'].isin(markers_of_interest)]

    print(df.shape)
    # if df.shape[0] >= 6000000:
    #     return
    print("Rescaling Politeness")
    df['Politeness'] = (df["Politeness"] - df['Politeness'].min())/(df['Politeness'].max() - df['Politeness'].min())
    

    bin_edges = load_2019_bin_edges(num_quantiles, markers_of_interest, columns_of_interest)
    print("Extracting Bins")
    ubins = get_bins(df[columns_of_interest], bin_edges, columns_of_interest)
    print("Getting bin names")
    df['bin'] = get_bin_names(ubins, namespace)

    # Compute extremeness
    print("Compute extremeness...")
    extremeness_cols = [col + "_Absolute" for col in columns_of_interest]
    df[extremeness_cols] = np.abs(df[columns_of_interest] - df[columns_of_interest].mean())

    columns_of_interest = columns_of_interest + extremeness_cols
    #TODO: REMOVE OR REWRITE
    Serialization.save_obj(df, f"stance_pipeline_full_data_with_sentences_{save_suffix}")

    print("Getting mean data")
    x = df.groupby("bin").mean()[columns_of_interest]
    print("Saving mean data")
    Serialization.save_obj(x, f"semantic_situation_mean_values_{num_quantiles}_{save_suffix}")
    print("Saving entire dataset")
    Serialization.save_obj(df[['subreddit', 'rel_marker', 'bin'] + columns_of_interest], f"stance_pipeline_full_data_{num_quantiles}_quantiles_{save_suffix}")



def get_nonzero_prop(df):
    """
    Return the number of non-zero elements in a dataframe, rounded to 2 digits.
    """
    print(np.round(np.count_nonzero(df)/df.size, 2))

    
def convert_to_cooc_matrix(data_suffix, num_quantiles):
    print("Loading data...")
    df = Serialization.load_obj(f"stance_pipeline_full_data_{num_quantiles}_quantiles_{data_suffix}")

    print("Loading markers...")
    all_markers = sorted(df['rel_marker'].unique())
    all_markers = [marker for marker in all_markers if marker not in ["'d", "10x"]]
    print("Num markers: ", len(all_markers))
    df = df[df['rel_marker'].isin(all_markers)]
    # Combine the subreddit and marker and aggregate
    print("Creating sub_markers...")
    df['sub_marker'] = df["subreddit"] + "__" + df['rel_marker']
    agg = df.groupby(["bin", "sub_marker"]).count()

    print("Creating all sub_marker combinations...")
    comms = df['subreddit'].unique()
    markers = df['rel_marker'].unique()
    bins = df['bin'].unique()
    com_markers = list(product(comms, markers))
    com_markers = ["__".join(pair) for pair in com_markers]

    # Create a matrix of all possible semantic situations and community markers with 0 values
    print("Creating full_matrix...")
    full_counts = pd.DataFrame(0, index=pd.MultiIndex.from_product([bins, com_markers], names=["bin", "sub_marker"]), columns=agg.columns)
    # Add to our attested matrix to fill in the blanks and get a full matrix
    print("Adding with existing_matrix...")
    total = agg.add(full_counts, fill_value=0)
    total = total.reset_index()
    # Create co-occurrence matrices
    print("Starting COOC Crosstab")
    cooc_matrix = pd.crosstab(total['bin'], total['sub_marker'], total['subreddit'], aggfunc="sum")
    print(f"Full Co-occurrence Matrix Size: {cooc_matrix.shape}")
    get_nonzero_prop(cooc_matrix)
    pav_matrix = pd.crosstab(df['subreddit'], df['rel_marker'])
    print(f"Pavalanathan Matrix Size: {pav_matrix.shape}")
    get_nonzero_prop(pav_matrix)

    pav_matrix = Serialization.save_obj(pav_matrix, f"pavalanathan_cooc_data_{data_suffix}") # change {full_data} to {intensifiers} to get subset
    cooc_matrix = Serialization.save_obj(cooc_matrix, f"our_cooc_data_{data_suffix}")



def generate_joint_distribution(save_suffix, num_quantiles, features, data_dir):
    
    markers_of_interest = load_intensifiers("../data/luo_intensifiers.txt")
    print(len(markers_of_interest))
    namespace = "VADPF"
    load_data_from_raw(data_dir, num_quantiles, save_suffix, markers_of_interest, namespace, features)
    convert_to_cooc_matrix(save_suffix, num_quantiles)


if __name__ == "__main__":
    generate_joint_distribution("luo_data_2019", 3, ["Valence", "Arousal", "Dominance", "Politeness", "Formality"], '/ais/hal9000/datasets/reddit/stance_pipeline/luo_data/full_features/')