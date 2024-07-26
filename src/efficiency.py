from utils.core_utils import *
from utils.data_utils import load_intensifiers, extract_relevant_markers, normalize_array, normalize_matrix
from utils.info_theory_utils import *
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import spearmanr, entropy
from scipy.special import entr
from collections import Counter
import seaborn as sns
from copy import deepcopy
from numpy.random import default_rng


class EfficiencyWrapper:
    def __init__(self,  encoder, needs, similarity_matrix):
        # self.encoder_df = encoder
        self.encoder = encoder#.to_numpy()
        self.pm = needs
        self.joint = self.encoder * self.pm
        self.similarity_matrix = similarity_matrix
        _, self.pw = margins(self.joint)
        self.num_referents = encoder.shape[0]
        self.num_words = encoder.shape[1]

        self.check_joint()
        self.check_encoder()
        
    def check_joint(self):
        np.testing.assert_approx_equal(self.joint.sum(), 1)
    
    def check_encoder(self):
        np.testing.assert_allclose(self.encoder.sum(axis=1).reshape(-1, ), np.ones(self.num_referents))

    def compute_qu_w(self):
        return self.joint/self.pw
    
    def compute_pu_w(self):
        qu_w = self.compute_qu_w()
        # CHECK THIS
        pu_w = np.dot(self.similarity_matrix, qu_w) # first column is p(u|w1), second is p(u|w2), third is p(u|w3), fourth is p(u|w4)
        x, y = np.where(qu_w == 0)
        pu_w[x, y] = 0
        return pu_w/pu_w.sum(axis=0)
    
    def communicative_cost(self, axis=None):
        pu_w = self.compute_pu_w()
        
        div = np.divide(1, pu_w, where=(pu_w!=0), out=np.zeros_like(pu_w))
        surprisal = np.log2(div, where=(pu_w!=0))

        # joint = self.encoder * need_probs
        info_val = (self.joint * surprisal).sum(axis=axis)
        return info_val  
    
    def situational_precision(self):
        pu_w = self.compute_pu_w()
        div = np.divide(1, pu_w, where=(pu_w!=0), out=np.zeros_like(pu_w))
        surprisal = np.log2(div, where=(pu_w!=0))

        # joint = self.encoder * need_probs
        info_val = (self.encoder * surprisal).sum(axis=1)
        return info_val  
    
    def full_precision_matrix(self):
        pu_w = self.compute_pu_w()
        div = np.divide(1, pu_w)#, where=(pu_w!=0), out=np.zeros_like(pu_w) + np.inf)
        surprisal = np.log2(div, where=(div!=0))
        return surprisal
    
    def situational_surprisal(self):
        pu_w = self.compute_pu_w()
        div = np.divide(1, pu_w, where=(pu_w!=0), out=np.zeros_like(pu_w))
        surprisal = np.log2(div, where=(pu_w!=0))

        # joint = self.encoder * need_probs
        info_val = (surprisal).sum(axis=1)
        return info_val  
   

    
    def complexity_style_shifting_joint(self, reddit_distribution):
        np.testing.assert_approx_equal(reddit_distribution.sum().sum(), 1)
        true_joint = self.joint.reshape(-1, )
        reddit_distribution_np = reddit_distribution.to_numpy().reshape(-1, )
        divs = DKL(true_joint, reddit_distribution_np)
        return divs



def generate_referent_similarity(save_suffix, variance, num_quantiles, distance_function, ptol):
    vadpf_reps = Serialization.load_obj(f"semantic_situation_mean_values_{num_quantiles}_{save_suffix}").to_numpy()
    squared_dist = squareform(pdist(vadpf_reps, "euclidean"))**2
    expd = np.exp(squared_dist * (-1/(2*variance)))
    return expd

def extract_neighbours(semantic_situations):
    number_codes = np.array([list(code[1::2]) for code in semantic_situations]).astype(int)
    row_to_neighbours = {}
    for i, code in enumerate(number_codes):
        dim_differences = np.abs(number_codes - code)
        neighbours = list(np.where(dim_differences.sum(axis=1) == 1)[0])
        extreme_neighbours = list(np.where((np.count_nonzero(dim_differences, axis=1) == 1) & (dim_differences.sum(axis=1) == 2))[0])
        row_to_neighbours[semantic_situations[i]] = [semantic_situations[j] for j in neighbours + extreme_neighbours]
    return row_to_neighbours

def generate_continuous_hypotheticals(vadpf_reps, curr_situations, sampled_situations, variance):
    curr_reps = vadpf_reps.loc[sampled_situations].to_numpy()
    total_reps = vadpf_reps.loc[curr_situations].to_numpy()
    squared_dist = cdist(total_reps, curr_reps, "euclidean")**2 #squareform(pdist(curr_reps, "euclidean"))**2
    expd = np.exp(squared_dist * (-1/(2*variance)))
    expd[expd<1e-5] = 0
    pw_u = expd/expd.sum(axis=1).reshape(-1, 1)
    return pw_u


def smooth_dataframe(df, delta):
    df += delta
    return normalize_matrix(df)



def generate_dataframe_mappings(data_suffix, num_intensifiers, delta_val, exclude_askreddit):

    df = Serialization.load_obj(f"our_cooc_data_{data_suffix}")
    if exclude_askreddit:
        df = df[[col for col in df.columns if not col.lower().startswith("askreddit__")]]

    cols = df.columns
    com_to_original_df = {}
    com_to_encoder = {}
    com_to_needs = {}

    marker_to_df = {}
    num_coms = df.shape[1]//num_intensifiers



    for i in range(num_coms):
        per_com_df = df[cols[i*num_intensifiers:(i+1)*num_intensifiers]]
        com_name = per_com_df.columns[0][ :per_com_df.columns[0].rindex("__")]
        per_com_df.columns = [col[col.rindex("__") + 2: ] for col in per_com_df.columns]
        assert len(per_com_df.columns) == num_intensifiers
        com_to_original_df[com_name.lower()] = per_com_df.copy()
        per_com_df = smooth_dataframe(per_com_df, delta_val)
        pm, pw = margins(per_com_df)
        per_com_encoder = per_com_df/pm
        com_to_encoder[com_name.lower()] = per_com_encoder
        com_to_needs[com_name.lower()] = pm
        

    for i in range(num_intensifiers): 
        per_marker_idxs = np.arange(i, num_coms*num_intensifiers, num_intensifiers)
        assert len(per_marker_idxs) == num_coms
        
        per_marker_df = df[df.columns[per_marker_idxs]]
        marker_name = per_marker_df.columns[0][per_marker_df.columns[0].rindex("__") + 2:]
        per_marker_df.columns = [col[ :col.rindex("__")].lower() for col in per_marker_df.columns]
        marker_to_df[marker_name] = per_marker_df
    
    return com_to_original_df, com_to_encoder, com_to_needs, marker_to_df


def generate_reddit_marker_distributions(marker_to_df):
    
    all_dists_non_norm = []
    all_dists = []
    for marker in marker_to_df:
        # Compute frequency of marker in each semantic situation across all communities
        total_dist = marker_to_df[marker].sum(axis=1)
        total_dist.name = marker
        all_dists_non_norm.append(total_dist)

        total_dist = total_dist / total_dist.sum()
        all_dists.append(total_dist)

    reddit_marker_dist = pd.concat(all_dists_non_norm, axis=1)
    reddit_marker_dist = normalize_matrix(reddit_marker_dist)
    marginal_marker_dist = pd.concat(all_dists, axis=1)
    return reddit_marker_dist, marginal_marker_dist


def extract_nonzero_data(df, similarity_matrix):
    non_zero_columns = df.sum()[df.sum()>0].index
    non_zero_rows = np.nonzero((df.sum(axis=1)>0).values)[0]
    curr_df = df.iloc[non_zero_rows][non_zero_columns]
    curr_df = normalize_matrix(curr_df)
    curr_sims = similarity_matrix[non_zero_rows, :][:, non_zero_rows]

    return curr_df, curr_sims

def smooth_dataframe(df, delta):
    df += delta
    return normalize_matrix(df)


 
def generate_efficiency_objects(community_encoders, community_needs, similarity_matrix):
    com_to_eobj = {}
    for com in community_encoders:
        curr_object = EfficiencyWrapper(community_encoders[com].to_numpy(), community_needs[com], similarity_matrix)
        com_to_eobj[com] = curr_object
    return com_to_eobj



def compute_efficiency(com_objects, complexity_baseline):
    complexity_vals = []
    cost_vals = []
    coms = []
    for com in com_objects:
        com_object = com_objects[com]
        complexity = com_object.complexity_style_shifting_joint(complexity_baseline)
        cost = com_object.communicative_cost()

        complexity_vals.append(complexity)
        cost_vals.append(cost)
        coms.append(com)

    return complexity_vals, cost_vals, coms


def compute_need_difference(base_need_dists, comparison_needs):
    divs = []
    for com in comparison_needs:
        divs.append(DKL(comparison_needs[com], base_need_dists[com]))
    return divs


# def plot_efficiency(complexity_vals, cost_vals, title):
#     plt.scatter(complexity_vals, cost_vals)
#     plt.xlabel("Complexity")
#     plt.ylabel("Communicative Cost")
#     plt.title(title)


def compute_attested_tradeoff_vals(community_encoders, needs, similarity_matrix, complexity_baseline, need_name):
    eobjs = generate_efficiency_objects(community_encoders, {com: needs for com in community_encoders}, similarity_matrix)
    complexity_vals, cost_vals, coms = compute_efficiency(eobjs, complexity_baseline)
    return pd.DataFrame({"coms": coms, "complexity": complexity_vals, "cost": cost_vals, "need_name": need_name}).set_index("coms")


# Old code for unit tests
# Unit tests
def tests():
    test_encoder = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ])

    test_encoder = np.array([
        [1],
        [1],
        [1],
        [1]
    ]).reshape(-1, 1)

    test_pm = np.array([0.25, 0.25, 0.25, 0.25]).reshape(-1, 1)
    test_joint = test_encoder * test_pm
    test_joint_df = pd.DataFrame(test_joint, index=['scarlet', 'crimson', 'light_red', 'green'], columns = ['WORD'])

    test_u_u_sim = np.array([
        [1.0, 0.9, 0.8, 0.6], #u1
        [0.9, 1.0, 0.9, 0.6], #u2
        [0.8, 0.9, 1.0, 0.8], #u3
        [0.6, 0.6, 0.8, 1.0], #u4
    ])

    np.testing.assert_allclose(test_u_u_sim, test_u_u_sim.T)
    test_obj = EfficiencyWrapper(test_joint_df, test_joint_df, test_u_u_sim)
    test_obj.communicative_cost()


    


    # reddit_average = raw_marker_distribution.iloc[non_zero_rows][non_zero_columns]
    # reddit_average = normalize_matrix(reddit_average)
    # reddit_e = EfficiencyWrapper(raw_marker_distribution, reddit_average, curr_sims)
    # com_to_min_complexity_eobj[com] = reddit_e

    # coms.append(com)

#     plt.hist(efficiency_df['num_markers'])
# plt.xlabel("# of Markers")
# plt.ylabel("Frequency")
# plt.title(f"# of Markers per Community (n={len(com_to_df)})")
# plt.show()

# plt.hist(efficiency_df['num_situations'])
# plt.xlabel("# of Semantic Situations")
# plt.ylabel("Frequency")
# plt.title(f"# of Semantic Situations per Community (n={len(com_to_df)})")