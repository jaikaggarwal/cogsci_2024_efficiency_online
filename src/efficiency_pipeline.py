from efficiency import *
from scipy.stats import ttest_rel
import argparse


DATA_SUFFIX_FULL = 'luo_data_2019'
DATA_SUFFIX_SMALL = 'luo_data_2019_10k_sample_3_dim'
NUM_QUANTILES = 3
NUM_SITUATIONS = 2**NUM_QUANTILES
VARIANCE = 0.01
SMOOTHING_DELTA = 1e-10
MINIMUM_SITUATION_PERCENTAGE = 0.9
NUM_SAMPLE = 50

INTENSIFIERS = sorted(load_intensifiers("../data/luo_intensifiers.txt"))
NUM_INTENSIFIERS = len(INTENSIFIERS)

# Takes 4 seconds
GLOBAL_SIMILARITY_MATRIX = generate_referent_similarity(DATA_SUFFIX_SMALL, VARIANCE, NUM_QUANTILES, None, None)
GLOBAL_VADPF_REPS = Serialization.load_obj(f"semantic_situation_mean_values_{NUM_QUANTILES}_{DATA_SUFFIX_SMALL}")


need_type_to_label = {
        "com_specific": "Community-Specific",
        "reddit_general": "Reddit-General",
        "uniform": "Uniform"
    }


def generate_frequent_situations(community_to_data_mapper, reddit_marker_dist):
    """
    Find all semantic situations that appear a sufficient number of times
    in the dataset.
    """
    # the number of communities a semantic situation must appear in
    min_community_frequency = MINIMUM_SITUATION_PERCENTAGE * len(community_to_data_mapper)
    
    # Count the number of communities a situation appears in
    com_to_sit_counts = {}

    # List of semantic situations that appear a non-zero number of times
    non_zero_situations = []
    
    for com in community_to_data_mapper:
        curr_df = community_to_data_mapper[com]
        counts = curr_df.sum(axis=1)
        com_to_sit_counts[com] = counts

        # All semantic situations that appear non-zero number of times
        non_zero = counts[counts!=0].index.tolist()
        non_zero_situations.extend(non_zero)

    # Currently unused, but can be used in the future
    sit_df = pd.DataFrame(com_to_sit_counts)

    # Number of communities each semantic situation appears in
    non_zero_com_counts = pd.DataFrame.from_dict(dict(Counter(non_zero_situations)), orient='index')
    valid_situations = non_zero_com_counts[non_zero_com_counts[0]>=min_community_frequency].index
    valid_situations = sorted(valid_situations)

    valid_situations_idxs = []
    for sit in valid_situations:
        valid_situations_idxs.append(reddit_marker_dist.index.get_loc(sit))
    return valid_situations, valid_situations_idxs

    




def generate_uniform_encoders(community_to_data_mapper, com_to_need_mapper, valid_situations):
    
    com_to_sampled_encoder = {}
    full_counts = pd.DataFrame(0, index=valid_situations, columns=INTENSIFIERS)
    
    # Create new encoders by sampling data from original
    for com in tqdm(community_to_data_mapper):
        if com not in com_to_need_mapper:
            continue
        sit_to_markers = {}
        curr = community_to_data_mapper[com]

        # Iterate through each semantic situation
        for i, row in curr.iterrows():
            # If the semantic situation has less data than we require, take all the data
            if row.sum() < NUM_SAMPLE:
                sit_to_markers[row.name] = row.to_dict()
            # Otherwise, sample 50 intensifier uses, weighted by how much they occur in the community overall (to avoid oversampling highly frequent words)
            else:
                sit_to_markers[row.name] = Counter(np.random.choice(INTENSIFIERS, size=NUM_SAMPLE, replace=True, p=row/row.sum()))
        
        # Only keep the situations that we know we intend to use
        per_com_df = pd.DataFrame(sit_to_markers).fillna(0).T.loc[valid_situations]
        per_com_df = full_counts.add(per_com_df, fill_value=0)
        per_com_df = smooth_dataframe(per_com_df, SMOOTHING_DELTA)
        pm, pw = margins(per_com_df)
        per_com_encoder = per_com_df/pm
        com_to_sampled_encoder[com.lower()] = per_com_encoder
    
    return com_to_sampled_encoder


def renormalize_needs(com_to_need_mapper, idxs):
    com_to_sampled_needs = {}
    for com in com_to_need_mapper:
        com_to_sampled_needs[com] = normalize_array(com_to_need_mapper[com][idxs])
        np.testing.assert_approx_equal(com_to_sampled_needs[com].sum(), 1)
    return com_to_sampled_needs


def adjust_similarity_matrix(current_similarity_matrix, idxs):
    return current_similarity_matrix[idxs][:, idxs]



def generate_efficiency_summary(encoders, need_type_to_need_probs, similarity_matrix, complexity_baseline):

    com_specific_needs = need_type_to_need_probs['com_specific']
    reddit_general_needs = need_type_to_need_probs['reddit_general']

    need_type_to_efficiency_objs = {need_type: {} for need_type in need_type_to_need_probs}
    efficiency_stats = {}

    for need_type in tqdm(need_type_to_need_probs):
        
        curr_needs = need_type_to_need_probs[need_type]
        eobjs = generate_efficiency_objects(encoders, curr_needs, similarity_matrix)
        complexity_vals, cost_vals, coms = compute_efficiency(eobjs, complexity_baseline)

        # if need_type in ['com_specific', 'reddit_general']:
        #     kl_from_baseline = compute_need_difference(reddit_general_needs, com_specific_needs)
        # else:
        #     kl_from_baseline = compute_need_difference(curr_needs, com_specific_needs)

        
        need_type_to_efficiency_objs[need_type]['objects'] = eobjs
        
        efficiency_stats[f'{need_type}_complexity'] = complexity_vals
        efficiency_stats[f'{need_type}_cost'] = cost_vals
        efficiency_stats[f'{need_type}_communities'] = coms
        # efficiency_stats[f'{need_type}_kl_from_baseline'] = kl_from_baseline

    
    efficiency_df = pd.DataFrame(efficiency_stats, index=efficiency_stats[f'com_specific_communities'])
    return need_type_to_efficiency_objs, efficiency_df



def visualize_single_need_tradeoff(xs, ys, colouration, need_type, seed, prefix, use_same_axes=False):
    plt.scatter(xs, ys, c=colouration, marker="o")
    plt.xlabel("Complexity")
    plt.ylabel("Communicative Cost")
    if use_same_axes:
        plt.xlim(0, 5)
        plt.ylim(6, 8)
    plt.title(f"Efficiency Tradeoff with \n{need_type_to_label[need_type]} Needs (n={len(xs)})")
    plt.colorbar()
    plt.savefig(f"/u/jai/efficiency/images/single_need_tradeoffs/{prefix}_{need_type}_seed_{seed}.png")
    plt.clf()


def visualize_multiple_need_tradeoffs(efficiency_df, need_type_to_need_probs, seed, prefix):
    for complexity_need_type in need_type_to_need_probs:
        xs = efficiency_df[f'{complexity_need_type}_complexity']
        for cost_need_type in need_type_to_need_probs:
            plt.scatter(xs, efficiency_df[f'{cost_need_type}_cost'],  label=need_type_to_label[cost_need_type], alpha=0.7)
        plt.xlabel("Complexity", fontsize=15)
        plt.ylabel("Communicative Cost", fontsize=15)
        plt.title(f"Efficiency Tradeoff with {need_type_to_label[complexity_need_type]}\nComplexity Across Needs (n={efficiency_df.shape[0]})", fontsize=15)
        plt.legend(loc="lower left")
        plt.savefig(f"/u/jai/efficiency/images/multiple_need_tradeoffs/{prefix}_{complexity_need_type}_complexity_seed_{seed}.png")
        plt.clf()



def visualization_wrapper(efficiency_df, need_type_to_need_probs, seed, prefix):
    for need_type in need_type_to_need_probs:
        visualize_single_need_tradeoff(efficiency_df[f'{need_type}_complexity'], 
                                       efficiency_df[f'{need_type}_cost'], 
                                       colouration=efficiency_df[f'{need_type}_kl_from_baseline'], 
                                       need_type=need_type, seed=seed, prefix=prefix)
        
    visualize_multiple_need_tradeoffs(efficiency_df, need_type_to_need_probs, seed, prefix)


def perform_statistical_tests():
    ttest_rel(efficiency_df['cost'], efficiency_df[f'reddit_cost'])



def compute_correlations(com_to_encoder_mapping, com_to_need_mapping, similarity_matrix, reddit_marker_dist, efficiency_df, seed, prefix):
    corrs = []
    sigs = []
    coms = []
    sits = []
    for com_of_interest in tqdm(com_to_encoder_mapping):
        new_need_eobjs = generate_efficiency_objects(com_to_encoder_mapping, {com: com_to_need_mapping[com_of_interest] for com in com_to_encoder_mapping}, similarity_matrix)
        AXIS = 1

        maximum_need_difference_situation_idx = com_to_need_mapping[com_of_interest].argmax()
        maximum_need_difference_situation = reddit_marker_dist.iloc[maximum_need_difference_situation_idx].name

        marker_counts = []
        situation_cost_counts = []
        com_to_marker_idx = {}

        for i, com in enumerate(efficiency_df.index):
            com_to_marker_idx[com] = i
            
            a = new_need_eobjs[com].encoder_df.loc[maximum_need_difference_situation].round(10)
            num_markers = a[a>0].shape[0]
            marker_counts.append(num_markers)

            situation_cost = new_need_eobjs[com].communicative_cost(axis=AXIS)[maximum_need_difference_situation_idx]
            situation_cost_counts.append(situation_cost)
        
        corr, sig = spearmanr(marker_counts, situation_cost_counts) 
        corrs.append(corr)
        sigs.append(sig)
        coms.append(com_of_interest)
        sits.append(maximum_need_difference_situation)

    df = pd.DataFrame({"corrs": corrs, "sigs": sigs, "coms": coms, "sits": sits})
    df.to_csv(f"/u/jai/efficiency/output/efficiency_stats/{prefix}_marker_cost_correlations_seed_{seed}.csv")
    



def main(seed):
    np.random.seed(seed)

    print("Loading original data...")
    com_to_original_df, com_to_encoder, com_to_needs, marker_to_df = generate_dataframe_mappings(DATA_SUFFIX_SMALL, NUM_INTENSIFIERS, SMOOTHING_DELTA)
    reddit_marker_dist, marginal_marker_dist = generate_reddit_marker_distributions(marker_to_df)
    reddit_need_probs, reddit_word_probs = margins(reddit_marker_dist)
    situation_to_neighbour = extract_neighbours(semantic_situations=reddit_marker_dist.index)
    uniform_need_probs = normalize_array(np.ones((len(reddit_need_probs), 1)))

    # Complete main analyses for original encoders
    
    # Step 1: Generate uniform, Reddit-general, and community-specific objects
    print("Generating original efficiency data frame...")
    need_type_to_need_probs = {
        "com_specific": com_to_needs,
        "reddit_general": {com: reddit_need_probs for com in com_to_needs},
        "uniform": {com: uniform_need_probs for com in com_to_needs}
    }
    all_need_probs = {curr: {com: com_to_needs[curr] for com in com_to_needs} for curr in com_to_needs}
    need_type_to_need_probs.update(all_need_probs)
    print(len(need_type_to_need_probs))

    need_to_eobjs_original, efficiency_df = generate_efficiency_summary(com_to_encoder, need_type_to_need_probs, GLOBAL_SIMILARITY_MATRIX, reddit_marker_dist)
    efficiency_df.to_csv(f"/u/jai/efficiency/output/efficiency_stats/original_encoders_seed_{seed}.csv")
    
    # visualization_wrapper(efficiency_df, need_type_to_need_probs, seed, prefix="original_encoders")
    # compute_correlations(com_to_encoder, com_to_needs, GLOBAL_SIMILARITY_MATRIX, reddit_marker_dist, efficiency_df, seed, prefix="original_encoders")

    # Repeat analyses with uniformly sampled encoders
    
    print("Loading full data...")
    # Add no smoothing here, since we just want the raw counts
    com_to_original_df_full, com_to_encoder_full, com_to_needs_full, marker_to_df_full = generate_dataframe_mappings(DATA_SUFFIX_FULL, NUM_INTENSIFIERS, 0)
    reddit_marker_dist_full, marginal_marker_dist_full = generate_reddit_marker_distributions(marker_to_df_full)
    reddit_need_probs_full, reddit_word_probs_full = margins(reddit_marker_dist_full)
    situation_to_neighbour_full = extract_neighbours(semantic_situations=reddit_marker_dist_full.index)

    # Extract situations with enough data
    frequent_situations, frequent_situations_idxs = generate_frequent_situations(com_to_original_df, reddit_marker_dist)
    
    # Renormalize distributions to account for fewer situations
    com_to_adjusted_needs = renormalize_needs(com_to_needs, frequent_situations_idxs)
    adjusted_similarity_matrix = adjust_similarity_matrix(GLOBAL_SIMILARITY_MATRIX, frequent_situations_idxs)
    adjusted_reddit_marker_dist = normalize_matrix(reddit_marker_dist.loc[frequent_situations])
    adjusted_reddit_need_probs = normalize_array(reddit_need_probs[frequent_situations_idxs])
    adjusted_uniform_need_probs = normalize_array(np.ones((len(frequent_situations), 1)))
    
    # Generate uniform encoders
    print("Sampling new encoders...")
    com_to_sampled_encoder = generate_uniform_encoders(com_to_original_df_full, com_to_needs, frequent_situations)
    
    print("Generating sampled efficiency data frame...")
    need_type_to_need_probs = {
        "com_specific": com_to_adjusted_needs,
        "reddit_general": {com: adjusted_reddit_need_probs for com in com_to_needs},
        "uniform": {com: adjusted_uniform_need_probs for com in com_to_needs}
    }
    all_need_probs = {curr: {com: com_to_adjusted_needs[curr] for com in com_to_adjusted_needs} for curr in com_to_adjusted_needs}
    need_type_to_need_probs.update(all_need_probs)
    need_to_eobjs_sampled, sampled_efficiency_df = generate_efficiency_summary(com_to_sampled_encoder, need_type_to_need_probs, adjusted_similarity_matrix, adjusted_reddit_marker_dist)
    sampled_efficiency_df.to_csv(f"/u/jai/efficiency/output/efficiency_stats/sampled_encoders_seed_{seed}.csv")
    # visualization_wrapper(sampled_efficiency_df, need_type_to_need_probs, seed, prefix="sampled_encoders")
    # compute_correlations(com_to_sampled_encoder, com_to_adjusted_needs, adjusted_similarity_matrix, adjusted_reddit_marker_dist, sampled_efficiency_df, seed, prefix="sampled_encoders")

    
   



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    
    args = parser.parse_args()
    main(args.seed)