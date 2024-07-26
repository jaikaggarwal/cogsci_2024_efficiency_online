from efficiency import *
from scipy.stats import ttest_rel, ttest_ind
from copy import deepcopy
from evolutionary_algorithm import * 
from multiprocessing import Pool
from pathlib import Path


DATA_SUFFIX = 'temporal_analysis_2017'
NUM_QUANTILES = 3
NUM_SITUATIONS = 2**NUM_QUANTILES
VARIANCE = 0.01
SMOOTHING_DELTA = 1e-10

INTENSIFIERS = sorted(load_intensifiers("../data/luo_intensifiers.txt"))
NUM_INTENSIFIERS = len(INTENSIFIERS)

# Takes 4 seconds
GLOBAL_SIMILARITY_MATRIX = generate_referent_similarity(DATA_SUFFIX, VARIANCE, NUM_QUANTILES, None, None)
GLOBAL_VADPF_REPS = Serialization.load_obj(f"semantic_situation_mean_values_{NUM_QUANTILES}_{DATA_SUFFIX}")


class HypotheticalObj:

    def __init__(self, variance, id, dist):
        self.variance = variance
        self.id = id
        self.dist = dist



def sample_gaussian_means_with_replacement():
    sampled_situations = np.random.choice(GLOBAL_VADPF_REPS.index, size=252, replace=True)
    sampled_centers = GLOBAL_VADPF_REPS.loc[sampled_situations].to_numpy()
    return sampled_centers

def sample_gaussian_variance(distribution_parameters):
    system_mean = distribution_parameters['system_mean']
    system_variance = distribution_parameters['system_variance']
    variance = np.abs(np.random.normal(system_mean, system_variance, size=252))
    return variance

def generate_encoder_base(distribution_parameters):
    sampled_means = sample_gaussian_means_with_replacement()
    sampled_variance = sample_gaussian_variance(distribution_parameters)
    squared_dist = cdist(GLOBAL_VADPF_REPS, sampled_means, "euclidean")**2
    original_expd = np.exp(squared_dist * (-1/(2*sampled_variance)))
    return original_expd


def generate_word_probs():
    word_prob_range = np.linspace(1, 1000, 100)
    sample_probs = 1/word_prob_range
    sample_probs /= sample_probs.sum()
    word_probs = np.random.choice(word_prob_range, size=252, replace=True, p=sample_probs)
    return normalize_array(word_probs)

def normalize_encoder(enc):
    return enc/enc.sum(axis=1).reshape(-1, 1)
    
def align_encoder(dist):
    idxs = dist.sum(axis=0).argsort()
    aligned = dist[:, idxs]
    return normalize_encoder(aligned)


def generate_continuous_hypothetical(distribution_parameters, num_systems):
    hypotheticals = []
    for _ in range(num_systems):
        original_expd = generate_encoder_base(distribution_parameters)
        word_probs = generate_word_probs()

        expd = original_expd * word_probs
        aligned = align_encoder(expd)
        hypotheticals.append(aligned)
    
    return hypotheticals



def generate_min_complexity_hypotheticals(min_variance, max_variance, num_steps, system_mean, num_systems_per_variance):
    np.random.seed(42)
    all_hyps = []
    for variance in tqdm(np.linspace(min_variance, max_variance, num_steps)):
        distribution_parameters = {"type": "gaussian", "system_mean": system_mean, "system_variance": variance}
        hyp_dists = generate_continuous_hypothetical(distribution_parameters, num_systems=num_systems_per_variance)
        ids = [f"gaussian___{np.round(variance, 6)}___{i}" for i in range(num_systems_per_variance)]
        hyps = [HypotheticalObj(variance, curr_id, curr_dist) for curr_id, curr_dist in zip(ids, hyp_dists)]
        all_hyps.extend(hyps)
    return all_hyps
           


def load_base_systems(load_presaved=True):

    if load_presaved:
        base_systems = Serialization.load_obj("base_gaussian_hypotheticals_0.00004_to_0.04_1000_steps")
    else:

        base_systems = generate_min_complexity_hypotheticals(min_variance=0.00004, 
                                                        max_variance=0.04, 
                                                        num_steps=1000, 
                                                        system_mean=0, 
                                                        num_systems_per_variance=10)
        Serialization.save_obj(base_systems, "base_gaussian_hypotheticals_0.00004_to_0.04_1000_steps")
    
    return base_systems



def swap_wrapper(base_hyp, num_system_swaps, num_column_swaps, include_base=True):
    all_hyps = []
    if include_base:
        all_hyps.append(base_hyp)
        
    base_dist = base_hyp.dist
    for i in range(num_system_swaps):
        base_dist = swap_columns(deepcopy(base_dist), num_swaps=num_column_swaps)
        new_id = f"{base_hyp.id}__{(i+1)*num_column_swaps}_swaps"
        new_hyp = HypotheticalObj(variance=base_hyp.variance, id=new_id, dist=deepcopy(base_dist))
        all_hyps.append(new_hyp)
    return all_hyps


def swap_columns(encoder, num_swaps):
    for i in range(num_swaps):
        idxs = np.random.choice(encoder.shape[1], 2, )
        encoder[:, idxs[::-1]] = encoder[:, idxs]
    return encoder


def generate_all_hypotheticals():
    base_systems = load_base_systems(load_presaved=True)

    np.random.seed(42)
    swapped_hyps = []
    for base_system in tqdm(base_systems):
        num_column_swaps = np.random.choice(100)
        swaps = swap_wrapper(base_system, num_system_swaps=1, num_column_swaps=num_column_swaps, include_base=False)
        swapped_hyps.extend(swaps)
    
    return swapped_hyps

    


def get_batches(arr, num_batches):
    # num_batches = len(arr) // batch_size
    batch_size = len(arr) // num_batches
    batches = []
    for i in range(num_batches+1):
        batches.append(arr[i*batch_size: (i+1)*batch_size])
    assert len(flatten(batches)) == len(arr)
    return batches


def compute_attested_tradeoff_vals_wrapper(batch, community_encoders, community_needs, similarity_matrix, complexity_dist, save_name):
    all_data = []
    for need in tqdm(batch):
        # TODO: REMOVE
        if need == 'uniform':
            continue
        all_data.append(compute_attested_tradeoff_vals(community_encoders, community_needs[need], similarity_matrix, complexity_dist, need))
    all_data = pd.concat(all_data)
    Serialization.save_obj(all_data, save_name)



def main(num_batches):

    # Load encoders + community needs
    print("Loading dataframes...")
    _, com_to_encoder, com_to_needs, _ = generate_dataframe_mappings(DATA_SUFFIX, NUM_INTENSIFIERS, SMOOTHING_DELTA, exclude_askreddit=False)

    # Add uniform need probabilities
    com_to_needs['uniform'] = normalize_array(np.ones((len(GLOBAL_SIMILARITY_MATRIX), 1)))

    # Complexity baseline
    complexity_baseline = com_to_encoder['askreddit'] * com_to_needs['askreddit']
    sorted_complexity_baseline = complexity_baseline[complexity_baseline.columns[complexity_baseline.sum(axis=0).argsort()]]

    # Remove r/AskReddit from analyses
    print("Dropping r/AskReddit")
    del com_to_encoder['askreddit']
    assert len(com_to_encoder) == len(com_to_needs) - 2 # no askreddit, no uniform

    # Load hypotheticals
    print("Loading Hypotheticals")
    hypothetical_objects = generate_all_hypotheticals()
    hyp_com_to_encoder = {hyp.id: pd.DataFrame(hyp.dist) for hyp in hypothetical_objects}

    # Create batches so we can do analyses on parts of the data
    batches = get_batches(list(com_to_needs.keys()), num_batches=num_batches)

    for i, batch in enumerate(batches[1:]):

        print("Computing efficiency of attested systems...")
        attested_save_name = f"{DATA_SUFFIX}_attested_efficiency_values_batch_{i}"
        compute_attested_tradeoff_vals_wrapper(batch, com_to_encoder, com_to_needs, GLOBAL_SIMILARITY_MATRIX, complexity_baseline, attested_save_name)

        print("Computing efficiency of hypothetical systems...")
        hypothetical_save_name = f"{DATA_SUFFIX}_hypothetical_efficiency_values_batch_{i}"
        compute_attested_tradeoff_vals_wrapper(batch, hyp_com_to_encoder, com_to_needs, GLOBAL_SIMILARITY_MATRIX, sorted_complexity_baseline, hypothetical_save_name)




def generate_fronts(all_encoders):
    objective_model = ObjectiveFunctions(None, None, None, None, objectives=['complexity', 'cost'], signs=[-1, -1])
    phe = PostHocEvolution(all_encoders, objective_model)

    fronts = phe.fast_non_dominated_sort(phe.population, len(phe.population))
    agent_to_true_rank = {}
    for i, front in enumerate(fronts):
        for agent in front:
            agent_to_true_rank[agent.name] = i + 1
    assert len(agent_to_true_rank) == all_encoders.shape[0]

    all_encoders['true_rank'] = all_encoders['name'].apply(lambda x: agent_to_true_rank[x])
    need_name = all_encoders['need_name'].iloc[0]
    Serialization.save_obj(all_encoders, f"{DATA_SUFFIX}_efficiency_scores/{need_name}")



def compute_fronts(num_batches, num_cores):

    for batch in range(num_batches + 1)[:1]:
        attested_save_name = f"{DATA_SUFFIX}_attested_efficiency_values_batch_{batch}"
        hypothetical_save_name = f"{DATA_SUFFIX}_hypothetical_efficiency_values_batch_{batch}"
        
        attested_mega_df = Serialization.load_obj(attested_save_name)
        hypothetical_mega_df = Serialization.load_obj(hypothetical_save_name)

        attested_mega_df['name'] = attested_mega_df.index
        hypothetical_mega_df['name'] = hypothetical_mega_df.index
        
        need_names = attested_mega_df['need_name'].unique()
        all_combined = []
        for need_name in tqdm(need_names):
            curr_attested = attested_mega_df[attested_mega_df['need_name'] == need_name]
            curr_hypothetical = hypothetical_mega_df[hypothetical_mega_df['need_name'] == need_name]
            curr_combined = pd.concat((curr_attested, curr_hypothetical))
            all_combined.append(curr_combined)  
        
        
        with Pool(num_cores) as p:
            r = list(tqdm(p.imap(generate_fronts, all_combined), total=len(all_combined)))
        


if __name__ == "__main__":
    main(num_batches=3)
    compute_fronts(num_batches=3, num_cores=24)
    
