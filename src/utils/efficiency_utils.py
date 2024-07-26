from .core_utils import *
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform


from .info_theory_utils import *
np.seterr(divide='ignore', invalid="ignore")


def reverse_annealing_wrapper(out_dir, pm, pu_m, init_qwm, betas, ptol=1e-8, max_iter=50):
    """
    true_pm: the community need probability
    pu_m: the probability of a given semantic situation under a particular meaning distribution
    init_qw_m: starting encoder that is maximally complex
    beta: the current beta value
    ctol: the threshold for determining convergence of the reverse annealing algorithm
    ptol: the threshold for removing words with low probabilities
    """
    curr_encoder = init_qwm

    for i in tqdm(range(len(betas))):
        # Initial joint is just the initial encoder multiplied by p(m)
        new_encoder, loop_count = reverse_annealing(pm, pu_m, curr_encoder, betas[i], ptol=ptol, max_iter=max_iter)
        with open(out_dir + f"variance_0.006_beta_{betas[i]}.pkl", "wb") as f:
            pickle.dump(new_encoder, f)
        curr_encoder = new_encoder
    return curr_encoder


def reverse_annealing(true_pm, pu_m, qw_m, beta, ctol=1e-3, ptol=1e-6, max_iter=50):
    """
    true_pm: the community need probability
    pu_m: the probability of a given semantic situation under a particular meaning distribution
    qw_m: the current encoder
    beta: the current beta value
    ctol: the threshold for determining convergence of the reverse annealing algorithm
    ptol: the threshold for removing words with low probabilities
    """
    # Threshold for determining convergence
    curr_encoder = qw_m
    # Starting value for the objective function
    prev_f = 100
    # This is how we compute the current encoder's tradeoff
    new_f = complexity_split(curr_encoder, true_pm) - beta * accuracy(curr_encoder, true_pm, pu_m)

    # Create p(x,y)
    curr_joint = pu_m * true_pm.reshape(-1, 1)
    np.testing.assert_almost_equal(curr_joint.sum(), 1)
    curr_pw = np.dot(curr_encoder.T, true_pm).T
    curr_mhat = m_hat(curr_encoder, true_pm, pu_m)
    
    loops = -1
    while np.abs(prev_f - new_f) >= ctol:
        loops += 1
        if loops >= max_iter:
            break

        # Let's first calculate the new encoder q(w|m)
        with Pool(processes=20) as p:
            input_objs = [[pu_m[i], curr_pw, beta, curr_mhat] for i in range(len(true_pm))]
            new_encoder_vals = list(p.imap(calculate_KL, input_objs))
        unnorm_encoder = np.stack(new_encoder_vals)
        new_encoder = (unnorm_encoder/unnorm_encoder.sum(axis=1).reshape(-1, 1))

        assert new_encoder.shape == curr_encoder.shape
        # Now we can calculate the new pw probabilities
        new_pw = np.dot(new_encoder.T, true_pm) # note that we use true pm, instead of the new pms
        # We drop any words that are used with probablility lower than 1e-5
        dropped = new_pw<=ptol # clusters to drop due to near-zero prob
        
        if any(dropped):
            new_pw = new_pw[~dropped] # drop ununsed clusters
            new_encoder = new_encoder[:, ~(dropped.reshape(-1))]
            # Renormalize
            new_encoder = (new_encoder/new_encoder.sum(axis=1).reshape(-1, 1))#np.multiply(qw_m,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # renormalize
        new_pw = np.dot(new_encoder.T, true_pm).T
        new_mhat = m_hat(new_encoder, true_pm, pu_m)

        curr_encoder = new_encoder
        curr_pw = new_pw
        curr_mhat = new_mhat

        prev_f = new_f
        new_f = complexity_split(curr_encoder, true_pm) - beta*accuracy(curr_encoder, true_pm, pu_m)

    return curr_encoder, loops


def reverse_annealing_variance_testing(pm, pu_m, init_qwm, betas, curr_var, ptol=1e-8, max_iter=50):
    OUTPUT_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/dec_6_full_run/variance_testing/"
    # encoder is conditional probability of w|m
    # pm is marginal probability
    curr_encoder = init_qwm
    # encoder_list = [curr_encoder]
    new_comp = []
    new_acc = []
    counts = []
    for i in tqdm(range(len(betas))):
        # Initial joint is just the initial encoder multiplied by p(m)
        new_encoder, loop_count = da(pm, pu_m, curr_encoder, betas[i], ptol=ptol, max_iter=max_iter)
        # encoder_list.append(new_encoder)
        with open(OUTPUT_DIR + f"variance_{curr_var}_beta_{betas[i]}.pkl", "wb") as f:
            pickle.dump(new_encoder, f)
        new_comp.append(complexity_split(new_encoder, pm))
        new_acc.append(accuracy(new_encoder, pm, pu_m))
        curr_encoder = new_encoder
        counts.append(loop_count)
    return new_comp, new_acc, counts





def compute_comp_efficiency(com):
    BASE_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/dec_6_full_run/community_efficiency/"
    variance_val = 0.006
    expd = np.exp(squared_dist * (-1/(2*variance_val)))
    curr_pu_m = expd/expd.sum(axis=1).reshape(-1, 1)
    
    # com_to_vals = {com: {} for com in communities}
    # with Pool(20) as p:
    #     r = list(tqdm(p.imap(mp_efficiency, [com for com in communities])))
    # for com in communities:
    ib = [ib for ib in all_ibs if ib.com_name == com][0]
    file_dir = BASE_DIR + com + "/"
    files = os.listdir(file_dir)
    sub_files = [(file, float(file[file.rindex("_")+1: -4])) for file in files]
    curr_files = [file[0] for file in sorted(sub_files, key=lambda x: x[1])]
    betas = [file[1] for file in sorted(sub_files, key=lambda x: x[1])]

    new_comps = []
    new_accs = []
    num_words = []
    for file in tqdm(curr_files):
        with open(file_dir + file, "rb") as curr_pickle:
            d = pickle.load(curr_pickle)
        new_comps.append(complexity_split(d, ib.p_m))
        new_accs.append(accuracy(d, ib.p_m, curr_pu_m))
        num_words.append(d.shape[1])
    sub_dict = {}
    sub_dict['comp'] = new_comps
    sub_dict['acc'] = new_accs
    sub_dict['num_words'] = num_words
    sub_dict['betas'] = betas
    # com_to_vals[com] = sub_dict
    Serialization.save_obj(sub_dict, f"efficiency_stats/{com}/{com}_efficiency_data")

    # Serialization.save_obj(com_to_vals, "efficiency_stats/all_community_efficiency_data")
