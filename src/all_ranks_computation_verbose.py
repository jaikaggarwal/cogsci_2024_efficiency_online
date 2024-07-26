from efficiency import *
from scipy.stats import ttest_rel, ttest_ind
from copy import deepcopy
from evolutionary_algorithm import PostHocEvolution, ObjectiveFunctions 
from multiprocessing import Pool
from pathlib import Path
import argparse



# Tagged with I if function is independent of current naming scheme.
# Additional tag of D added if need to verify current naming scheme matches what the function expects.

NUM_QUANTILES = 3
NUM_SITUATIONS = NUM_QUANTILES**5
VARIANCE = 0.01
SMOOTHING_DELTA = 1e-10
INTENSIFIERS = sorted(load_intensifiers("../data/luo_intensifiers.txt"))
NUM_INTENSIFIERS = len(INTENSIFIERS)



def load_intersected_communities():
    # REMOVE ALL NON TEMPORAL SUBREDDITS
    # Relevant Communities
    print("Keeping only temporal communities")
    with open("../data/temporal_intersection.txt", "r") as f:
        attested_communities = []
        for line in f:
            subreddit = line.strip().lower()
            if subreddit != "":
                attested_communities.append(subreddit)

    encoders_of_interest = set(attested_communities)
    print(f"Number of communities: {len(encoders_of_interest)}")
    return encoders_of_interest



class HypotheticalObj:

    def __init__(self, variance, id, dist):
        self.variance = variance
        self.id = id
        self.dist = dist



def sample_gaussian_means_with_replacement(vadpf_representations): #I
    sampled_situations = np.random.choice(vadpf_representations.index, size=252, replace=True)
    sampled_centers = vadpf_representations.loc[sampled_situations].to_numpy()
    return sampled_centers



def sample_gaussian_variance(distribution_parameters): #I
    system_mean = distribution_parameters['system_mean']
    system_variance = distribution_parameters['system_variance']
    variance = np.abs(np.random.normal(system_mean, system_variance, size=252))
    return variance



def generate_encoder_base(distribution_parameters, vadpf_representations): #I
    sampled_means = sample_gaussian_means_with_replacement(vadpf_representations) #I
    sampled_variance = sample_gaussian_variance(distribution_parameters) #I
    squared_dist = cdist(vadpf_representations, sampled_means, "euclidean")**2 #I
    original_expd = np.exp(squared_dist * (-1/(2*sampled_variance)))
    return original_expd



def generate_word_probs(): #I
    word_prob_range = np.linspace(1, 1000, 100)
    sample_probs = 1/word_prob_range
    sample_probs /= sample_probs.sum()
    word_probs = np.random.choice(word_prob_range, size=252, replace=True, p=sample_probs)
    return normalize_array(word_probs)



def normalize_encoder(enc): #I
    return enc/enc.sum(axis=1).reshape(-1, 1)
    


def align_encoder(dist): #I
    idxs = dist.sum(axis=0).argsort()
    aligned = dist[:, idxs]
    return normalize_encoder(aligned) #I



def generate_continuous_hypothetical(distribution_parameters, vadpf_representations, num_systems): #I
    hypotheticals = []
    for _ in range(num_systems):
        original_expd = generate_encoder_base(distribution_parameters, vadpf_representations) #I
        word_probs = generate_word_probs() #I

        expd = original_expd * word_probs
        aligned = align_encoder(expd) #I
        hypotheticals.append(aligned)
    
    return hypotheticals



def generate_min_complexity_hypotheticals(min_variance, max_variance, num_steps, system_mean, num_systems_per_variance, vadpf_representations): #I
    np.random.seed(42)
    all_hyps = []
    for variance in tqdm(np.linspace(min_variance, max_variance, num_steps)):
        distribution_parameters = {"type": "gaussian", "system_mean": system_mean, "system_variance": variance}
        hyp_dists = generate_continuous_hypothetical(distribution_parameters, vadpf_representations, num_systems=num_systems_per_variance) #I
        ids = [f"gaussian___{np.round(variance, 6)}___{i}" for i in range(num_systems_per_variance)]
        hyps = [HypotheticalObj(variance, curr_id, curr_dist) for curr_id, curr_dist in zip(ids, hyp_dists)]
        all_hyps.extend(hyps)
    return all_hyps
           


def swap_columns(encoder, num_swaps, random_generator): #I
    for i in range(num_swaps):
        idxs = random_generator.choice(encoder.shape[1], 2, )
        encoder[:, idxs[::-1]] = encoder[:, idxs]
    return encoder



def swap_wrapper(base_hyp, num_system_swaps, num_column_swaps, random_generator, include_base=True): #I
    all_hyps = []
    if include_base:
        all_hyps.append(base_hyp)
        
    base_dist = base_hyp.dist
    for i in range(num_system_swaps):
        base_dist = swap_columns(deepcopy(base_dist), num_swaps=num_column_swaps, random_generator=random_generator) #I
        new_id = f"{base_hyp.id}__{(i+1)*num_column_swaps}_swaps"
        new_hyp = HypotheticalObj(variance=base_hyp.variance, id=new_id, dist=deepcopy(base_dist))
        all_hyps.append(new_hyp)
    return all_hyps

    

def load_all_hypothetical_systems(load_presaved_swapped_systems=True, load_presaved_base_systems=True): #I

    if load_presaved_swapped_systems:
        print("Loading presaved swapped systems")
        return Serialization.load_obj(f"swapped_gaussian_hypotheticals_0.00004_to_0.04_1000_steps")

    print("Computing Hypotheticals from Scratch")
    if load_presaved_base_systems:
        print("Loading presaved hypotheticals")
        base_systems = Serialization.load_obj(f"base_gaussian_hypotheticals_0.00004_to_0.04_1000_steps")
    else:
        vadpf_representations = Serialization.load_obj(f"semantic_situation_mean_values_{NUM_QUANTILES}_cogsci_2024")
        base_systems = generate_min_complexity_hypotheticals(min_variance=0.00004, 
                                                        max_variance=0.04, 
                                                        num_steps=1000, 
                                                        system_mean=0, 
                                                        num_systems_per_variance=10, 
                                                        vadpf_representations=vadpf_representations) #I
        Serialization.save_obj(base_systems, f"base_gaussian_hypotheticals_0.00004_to_0.04_1000_steps")
    
   
    print("Swapping")
    random_generator = np.random.default_rng(42)
    swapped_hyps = []
    for base_system in tqdm(base_systems):
        num_column_swaps = random_generator.choice(100)
        swaps = swap_wrapper(base_system, num_system_swaps=1, num_column_swaps=num_column_swaps, random_generator=random_generator, include_base=False) #I
        swapped_hyps.extend(swaps)
    
    Serialization.save_obj(swapped_hyps, f"swapped_gaussian_hypotheticals_0.00004_to_0.04_1000_steps")
    
    return swapped_hyps



def get_batches(arr, num_batches): #I
    #num_batches = len(arr) // batch_size
    batch_size = (len(arr) // num_batches) + 1
    batches = []
    for i in range(num_batches):
        batches.append(arr[i*batch_size: (i+1)*batch_size])
    print(batches)
    assert len(flatten(batches)) == len(arr)
    return batches



def compute_attested_tradeoff_vals_wrapper(community_encoders, community_needs, similarity_matrix, complexity_dist, save_name): #D #I
    for need in tqdm(community_needs):
        curr_save_name = f"{save_name}/{need}"
        if not os.path.exists(DATA_OUT_DIR + curr_save_name + ".pkl"):
            curr_data = compute_attested_tradeoff_vals(community_encoders, community_needs[need], similarity_matrix, complexity_dist, need)
            Serialization.save_obj(curr_data, curr_save_name)



def generate_fronts(data): #D #I
    save_dir = data['save_dir']
    all_encoders = data['df']
    need_name = all_encoders['need_name'].iloc[0]

    if os.path.exists(DATA_OUT_DIR + save_dir + "/" + need_name + ".pkl"):
        return


    objective_model = ObjectiveFunctions(None, None, None, None, objectives=['complexity', 'cost'], signs=[-1, -1])
    phe = PostHocEvolution(all_encoders, objective_model)

    # print("FND SORT")
    fronts = phe.fast_non_dominated_sort(phe.population, len(phe.population)) #D #I
    agent_to_true_rank = {}
    for i, front in enumerate(fronts):
        for agent in front:
            agent_to_true_rank[agent.name] = i + 1
    assert len(agent_to_true_rank) == all_encoders.shape[0]

    # print("SAVING")

    all_encoders['true_rank'] = all_encoders['name'].apply(lambda x: agent_to_true_rank[x])
    Serialization.save_obj(all_encoders, f"{save_dir}/{need_name}")



def extract_realistic_systems():
    base_dir = f"temporal_analysis/needs_2019/complexity_2019/similarity_2019/"

    attested_df = Serialization.load_obj(base_dir + "attested_efficiency_values/encoders_2019/askreddit")
    hypothetical_df = Serialization.load_obj(base_dir + "hypothetical_efficiency_values/askreddit")

    max_complexity = attested_df['complexity'].max()
    realistic_hypothetical_df = hypothetical_df[hypothetical_df['complexity'] <= max_complexity]
    return realistic_hypothetical_df.index.tolist()
        


def compute_fronts(attested_save_dir, hypothetical_save_dir, ranks_save_dir, needs, rank_set, num_cores): #D #I

    print("Computing Fronts")
    all_combined = []
    realistic_communities = extract_realistic_systems()

    for need in needs:

        attested_df = Serialization.load_obj(f"{attested_save_dir}/{need}")
        hypothetical_df = Serialization.load_obj(f"{hypothetical_save_dir}/{need}")
        
        attested_df['name'] = attested_df.index
        hypothetical_df['name'] = hypothetical_df.index

        if rank_set == "attested":
            combined = attested_df
        elif rank_set == "realistic":
            hypothetical_df = hypothetical_df[hypothetical_df.index.isin(realistic_communities)]
            combined = pd.concat((attested_df, hypothetical_df)) 
        else:
            combined = pd.concat((attested_df, hypothetical_df)) 
        
        all_combined.append({
            "df": combined,
            "save_dir": ranks_save_dir
        })

        
    with Pool(num_cores) as p:
        r = list(tqdm(p.imap(generate_fronts, all_combined), total=len(all_combined))) #D #I
        


def load_2019_data():
    print("Loading 2019 data")
    suffix_2019 = 'cogsci_2024'
    _, com_to_encoder_2019, com_to_needs_2019, _ = generate_dataframe_mappings(suffix_2019, NUM_INTENSIFIERS, SMOOTHING_DELTA, exclude_askreddit=False) #D #I

    # Takes 4 seconds
    print("Generating Referent Similarity")   
    similarity_matrix_2019 = generate_referent_similarity(suffix_2019, VARIANCE, NUM_QUANTILES, None, None) #I
    print("Loading Mean Values")  
    vadpf_reps_2019 = Serialization.load_obj(f"semantic_situation_mean_values_{NUM_QUANTILES}_{suffix_2019}")

    return com_to_encoder_2019, com_to_needs_2019, similarity_matrix_2019, vadpf_reps_2019



def load_encoders(encoder_params):
    print(f"Loading encoders from {encoder_params['data_in_suffix']}...")
    _, com_to_encoder, _, _ = generate_dataframe_mappings(encoder_params['data_in_suffix'], NUM_INTENSIFIERS, SMOOTHING_DELTA, exclude_askreddit=False) #D #I
    # Removing all that are not in RELEVANT
    for com in list(com_to_encoder.keys()):
        if com not in encoder_params['encoders_of_interest']:
            del com_to_encoder[com]
        if com.lower() == 'askreddit':
            del com_to_encoder[com]
    return com_to_encoder



def load_needs(need_params):
    print(f"Loading needs from {need_params['data_in_suffix']}...")
    _, _, com_to_needs, _ = generate_dataframe_mappings(need_params['data_in_suffix'], NUM_INTENSIFIERS, SMOOTHING_DELTA, exclude_askreddit=False) #D #I
    for com in list(com_to_needs.keys()):
        if com not in need_params['needs_of_interest']:
            del com_to_needs[com]

     # Add uniform need probabilities
    if 'uniform' in need_params['needs_of_interest']:
        com_to_needs['uniform'] = normalize_array(np.ones((NUM_SITUATIONS, 1))) #I
    
    return com_to_needs




def load_complexity_baseline(complexity_params):
    _, com_to_encoder, com_to_needs, _ = generate_dataframe_mappings(complexity_params['data_in_suffix'], NUM_INTENSIFIERS, SMOOTHING_DELTA, exclude_askreddit=False) #D #I
    complexity_baseline = com_to_encoder['askreddit'] * com_to_needs['askreddit']
    sorted_complexity_baseline = complexity_baseline[complexity_baseline.columns[complexity_baseline.sum(axis=0).argsort()]] 
    return complexity_baseline, sorted_complexity_baseline   


def load_similarity_matrix(similarity_params):
    return generate_referent_similarity(similarity_params['data_in_suffix'], VARIANCE, NUM_QUANTILES, None, None)



def main(encoder_params, need_params, complexity_params, similarity_params, attested_save_name, hypothetical_save_name):
    # General parameters

    print("Loading dataframes...")

    com_to_encoder = load_encoders(encoder_params)
    com_to_needs = load_needs(need_params)
    complexity_baseline, sorted_complexity_baseline = load_complexity_baseline(complexity_params)
    similarity_matrix = load_similarity_matrix(similarity_params)

    print(f"Size of com_to_encoder: {len(com_to_encoder.keys())}")
    print(f"Size of com_to_needs: {len(com_to_needs.keys())}")

    print("Loading Hypotheticals")
    hypothetical_objects = load_all_hypothetical_systems() #I
    hyp_com_to_encoder = {hyp.id: pd.DataFrame(hyp.dist) for hyp in hypothetical_objects}

    

    print("Computing efficiency of attested systems...")
    compute_attested_tradeoff_vals_wrapper(com_to_encoder, com_to_needs, similarity_matrix, complexity_baseline, attested_save_name) #D #I

    print("Computing efficiency of hypothetical systems...")
    compute_attested_tradeoff_vals_wrapper(hyp_com_to_encoder, com_to_needs, similarity_matrix, sorted_complexity_baseline, hypothetical_save_name) #D #I



# TODO: Rewrite computing fronts to not work on batches
# TODO: Test that the values for the askreddit attested values are the same as the ones we used for cogsci
# TODO: Test that the uniform and askreddit complexity/cost scores for the 2019 encoders are the same as they were for cogsci
# TODO: Test that the uniform and askreddit ranks for the 2019 encoders are the same as they were for cogsci (finish hypotheticals)
 





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder_year", type=str)
    parser.add_argument("-needs", type=str, choices=['askreddit', 'uniform', 'general', 'community-specific', "all"])
    parser.add_argument("-needs_year", type=str)
    parser.add_argument("-complexity_year", type=str)
    parser.add_argument("-similarity_year", type=str)
    parser.add_argument("-rank_set", type=str)
    args = parser.parse_args()


    encoder_year_in_suffix = f"temporal_analysis_{args.encoder_year}_using_2019_bins" if args.encoder_year != "2019" else "cogsci_2024"
    needs_year_in_suffix = f"temporal_analysis_{args.needs_year}_using_2019_bins" if args.needs_year != "2019" else "cogsci_2024"
    complexity_year_in_suffix = f"temporal_analysis_{args.complexity_year}_using_2019_bins" if args.complexity_year != "2019" else "cogsci_2024"
    similarity_year_in_suffix = f"temporal_analysis_{args.similarity_year}_using_2019_bins" if args.similarity_year != "2019" else "cogsci_2024"


    encoder_year_out_suffix = f"encoders_{args.encoder_year}"
    needs_year_out_suffix = f"needs_{args.needs_year}"
    complexity_year_out_suffix = f"complexity_{args.complexity_year}"
    similarity_year_out_suffix = f"similarity_{args.similarity_year}"

    print(f"Encoder output suffix: {encoder_year_out_suffix}")
    print(f"Need output suffix: {needs_year_out_suffix}")
    print(f"Complexity output suffix: {complexity_year_out_suffix}")
    print(f"Similarity output suffix: {similarity_year_out_suffix}")


    communities_of_interest = load_intersected_communities()

    if args.needs == "askreddit":
        needs_of_interest = ["askreddit"]
    elif args.needs == "uniform":
        needs_of_interest = ["uniform"]
    elif args.needs == "general":
        needs_of_interest = ["uniform", "askreddit"]
    elif args.needs == "community-specific":
        needs_of_interest = list(communities_of_interest)
    elif args.needs == "all":
        needs_of_interest = list(communities_of_interest) + ["uniform", "askreddit"]

    encoder_params = {
        "encoders_of_interest": communities_of_interest,
        "data_in_suffix": encoder_year_in_suffix,
        "data_out_suffix": encoder_year_out_suffix
    }

    need_params = {
        "needs_of_interest": needs_of_interest,
        "data_in_suffix": needs_year_in_suffix,
        "data_out_suffix": needs_year_out_suffix
    }

    complexity_params = {
        "data_in_suffix": complexity_year_in_suffix,
        "data_out_suffix": complexity_year_out_suffix
    }

    similarity_params = {
        "data_in_suffix": similarity_year_in_suffix,
        "data_out_suffix": similarity_year_out_suffix
    }
    

    save_name_root = f"temporal_analysis/{need_params['data_out_suffix']}/{complexity_params['data_out_suffix']}/{similarity_params['data_out_suffix']}"
    attested_save_name = f"{save_name_root}/attested_efficiency_values/{encoder_params['data_out_suffix']}/"
    hypothetical_save_name = f"{save_name_root}/hypothetical_efficiency_values/"
    ranks_save_name = f"{save_name_root}/ranks_{args.rank_set}/{encoder_params['data_out_suffix']}/"
    

    make_directory(DATA_OUT_DIR + attested_save_name)
    make_directory(DATA_OUT_DIR + hypothetical_save_name)
    make_directory(DATA_OUT_DIR + ranks_save_name)

    print(f"Attested save name: {attested_save_name}")
    print(f"Hypothetical save name: {hypothetical_save_name}")
    print(f"Ranks save name: {ranks_save_name}")

    main(encoder_params, need_params, complexity_params, similarity_params, attested_save_name, hypothetical_save_name)
    compute_fronts(attested_save_name, hypothetical_save_name, ranks_save_name, needs_of_interest, args.rank_set, num_cores=24)


