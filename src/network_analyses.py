
from utils.core_utils import *

import time
from collections import defaultdict
import networkx as nx
from multiprocessing import Pool
import gc
from sklearn.metrics.pairwise import cosine_similarity

from itertools import permutations
from scipy.sparse import load_npz

# Fastest serializer
try:
    import orjson 
except:
    pass

# Fastest non-binary serializer
try:
    import ujson as json
except:
    import json

# DIRECTORIES
DATA = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/social_network_data/"
DICTIONARY_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/network_analysis/dictionaries/"
GRAPH_DIR = '/ais/hal9000/datasets/reddit/stance_pipeline/luo_data/network_analysis/graphs/'
CACHE = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/network_analysis/cache/"

attested_communities = pd.read_csv("../data/cogsci_2024_communities.txt", header=None)
attested_communities_set = set(attested_communities[0].tolist())

file_prefix = "RC_2019-"
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
all_files = [file_prefix + month for month in months]
print(all_files)


def write_dict_to_json(data, path):
    with open(path,"w") as f:
        json.dump(data, f)


def load_dict_from_json(path):
    with open(path,"r") as f:
        return json.load(f)




def generate_mapping(filename, test_mode=False):
    # Input must be in JSON readable format, containing fields: author, subreddit, parent_id
    # Generate Author:Subreddit dictionary
    
    # DATA STORAGE
    DICT =  defaultdict(lambda: defaultdict(int))
    TOP_LEVEL_DICT = defaultdict(lambda: defaultdict(int))

    # Setting input path
    path = f"{DATA}/{filename}.json"
    line_counter = 0
    print("Iterating over file : "+filename+"\n")

    # Iterating over each line in the file
    with open(path, "r") as file:
        for line in tqdm(file, total=100000000):
            
            # Load in one JSON object
            obj = json.loads(line)
            
            # Important values
            author = obj['author']
            subreddit = obj['subreddit']

            # Incrementing occurence
            DICT[author][subreddit] += 1

            if obj["is_top_level"]:
                TOP_LEVEL_DICT[author][subreddit] += 1
            
            line_counter += 1
            if test_mode and line_counter >= 5000000:
                break

    return dict(DICT), dict(TOP_LEVEL_DICT), filename


    


# READ IN MONTH BY MONTH
def condense(files, top_level_only=False):

    # GENERATING CONDENSED AUTHOR-SUBREDDIT DICTIONARY (COMBINE ALL 12 MONTHS INTO ONE DICTIONARY)
    CONDENSED = defaultdict(lambda: defaultdict(int))

    file_suffix = "_top.json" if top_level_only else ".json"
    for filename in tqdm(files):
        file_path = f"{DICTIONARY_DIR}/{filename}{file_suffix}"
        DICT = load_dict_from_json(file_path)
        
        total = 0
        # Iterate over all authors
        for author in DICT:
            # Add up subreddit interactions
            for subreddit in DICT[author]:                
                # Incrementing    
                CONDENSED[author][subreddit] += DICT[author][subreddit]
                total += DICT[author][subreddit]
    return CONDENSED

    

def compute_subreddit_size(activity_dict):
    # subreddit_post_dict = defaultdict(int)
    # for author in tqdm(activity_dict):
    #     for subreddit in activity_dict[author]:
    #         subreddit_post_dict[subreddit] += activity_dict[author][subreddit]
    subreddit_size_dict = defaultdict(int)
    for author in tqdm(activity_dict):
        for subreddit in activity_dict[author]:
            subreddit_size_dict[subreddit] += 1
    return subreddit_size_dict




def compute_author_activity(activity_dict, threshold):
    author_post_dict = defaultdict(int)
    for author in tqdm(activity_dict):
        total = 0
        for subreddit in activity_dict[author]:
            total += activity_dict[author][subreddit]
        if total >= threshold:
            author_post_dict[author] += total
    return author_post_dict



def compute_subreddit_loyalty(activity_dict, active_authors_dict, threshold):
    # We maintain an author_level loyalty dictionary for testing purposes
    author_loyalty_dict = {}
    subreddit_active_users_dict = defaultdict(int)
    subreddit_loyal_users_dict = defaultdict(int)

    for author in tqdm(activity_dict):
        if author not in active_authors_dict:
            continue

        # Find subreddit with maximum posts
        max_value = 0
        max_subreddit = None
        total = 0

        for subreddit in activity_dict[author]:

            # Add one to the subreddit's total number of users
            subreddit_active_users_dict[subreddit] += 1

            # Find subreddit with maximum posts
            val = activity_dict[author][subreddit]
            if val > max_value:
                max_value = val
                max_subreddit = subreddit
            total += val

        # Check if author has more than loyalty_threshold posts in subreddit
        if (max_value/total) >= threshold:
            subreddit_loyal_users_dict[max_subreddit] += 1
            author_loyalty_dict[author] = max_subreddit

    subreddit_loyalty_dict = {}
    for subreddit in subreddit_active_users_dict:
        num_loyal = subreddit_loyal_users_dict[subreddit]
        num_active = subreddit_active_users_dict[subreddit]
        subreddit_loyalty_dict[subreddit] = num_loyal/num_active
    
    return author_loyalty_dict, subreddit_loyalty_dict




def extract_top_active_authors(author_post_dict, threshold):
    values = list(author_post_dict.values())
    num_posts_at_threshold = np.quantile(values, 1-threshold)
    print(num_posts_at_threshold)

    top_authors = {}
    for author in tqdm(author_post_dict):
        if author_post_dict[author] >= num_posts_at_threshold:
            top_authors[author] = author_post_dict[author]
    return top_authors


def generate_subreddit_author_metadata(data_dict):

    filename = data_dict['filename']
    active_authors = data_dict['active_authors']
    relevant_communities = data_dict['relevant_communities']

    # DATA STORAGE
    SUBREDDIT_TO_AUTHOR_TO_PARENT_IDS =  defaultdict(lambda: defaultdict(list))

    # Setting input path
    path = f"{DATA}/{filename}.json"
    line_counter = 0

    print("Iterating over file : "+filename+"\n")

    with open(path, "r") as file:
        # Iterating over each line in the file
        for line in tqdm(file, total=100000000):
            
            # Load in one JSON object
            obj = json.loads(line)

            # Only keep top active authors
            author = obj['author']
            if author not in active_authors:
                continue
            
            # Only keep subreddits we care about
            subreddit = obj['subreddit']
            if subreddit not in relevant_communities:
                continue

            # We don't need r/AskReddit
            if subreddit == "AskReddit":
                continue

            # We can't make parent_links for top-level comments
            if obj['is_top_level']:
                continue

            parent_id = obj['parent_id'][3:]
            SUBREDDIT_TO_AUTHOR_TO_PARENT_IDS[subreddit][author].append(parent_id)

            line_counter += 1
            # if line_counter >= 1000000:
            #     break

        
    return dict(SUBREDDIT_TO_AUTHOR_TO_PARENT_IDS), filename



def generate_density_id_metadata(filename):

    # DATA STORAGE
    ID_TO_AUTHOR = {}

    # Setting input path
    path = f"{DATA}/{filename}.json"
    line_counter = 0

    print("Iterating over file : "+filename+"\n")

    with open(path, "r") as file:
        # Iterating over each line in the file
        for line in tqdm(file, total=100000000):
            
            # Load in one JSON object
            obj = json.loads(line)

            # # Only keep top active authors
            # author = obj['author']
            # if author not in active_authors:
            #     continue
            
            # Only keep subreddits we care about
            subreddit = obj['subreddit']
            if subreddit not in attested_communities_set:
                continue

            # We don't need r/AskReddit
            if subreddit == "AskReddit":
                continue

            # Track who wrote this post
            curr_id = obj['id']
            author = obj['author']
            ID_TO_AUTHOR[curr_id] = author

            # line_counter += 1
            # if line_counter >= 1000000:
            #     break

    write_dict_to_json(ID_TO_AUTHOR,  f"{DICTIONARY_DIR}/{filename}_id_to_author.json")



def load_density_metadata(files, active_authors):

    subreddit_to_author_to_parent_ids = defaultdict(lambda: defaultdict(list))
    id_to_author = {}

    for file in tqdm(files):
        curr_subreddit_to_author_to_parent_ids = load_dict_from_json(f"{DICTIONARY_DIR}/{file}_author_to_parent_ids.json")
        for subreddit in curr_subreddit_to_author_to_parent_ids:
            for author in curr_subreddit_to_author_to_parent_ids[subreddit]:
                curr_ids = curr_subreddit_to_author_to_parent_ids[subreddit][author]
                subreddit_to_author_to_parent_ids[subreddit][author].extend(curr_ids)

        curr_id_to_author = load_dict_from_json(f"{DICTIONARY_DIR}/{file}_id_to_author.json")
        for id in curr_id_to_author:
            if curr_id_to_author[id] in active_authors:
                id_to_author[id] = curr_id_to_author[id]
    
    return subreddit_to_author_to_parent_ids, id_to_author





def generate_density_edges(subreddit_author_parent_id_dict, id_to_author_dict):
    for subreddit in tqdm(subreddit_author_parent_id_dict):
        edges = defaultdict(list)
        for author in subreddit_author_parent_id_dict[subreddit]:
            for parent_id in subreddit_author_parent_id_dict[subreddit][author]:
                if parent_id in id_to_author_dict:
                    parent_author = id_to_author_dict[parent_id]
                    edges[author].append(parent_author)
        # Save current set of edges
        Serialization.save_obj(edges, f"cogsci_2024_density/{subreddit}")
        

def compute_density(relevant_communities):
    subreddit_to_density = {}
    relevant_communities = relevant_communities.difference({"AskReddit"})
    for subreddit in tqdm(relevant_communities):
        adjacency_list = Serialization.load_obj(f"cogsci_2024_density/{subreddit}")
        curr_graph = nx.Graph(adjacency_list)
        subreddit_to_density[subreddit] = nx.function.density(curr_graph)
    return subreddit_to_density



def compute_user_overlap(activity_dict, ROWS):

    # Initialization
    user_overlap = np.zeros((len(ROWS), len(ROWS)))

    # Iterating over all authors
    for author in activity_dict:
        
        # Extract the subreddits author has posted in
        subreddits = activity_dict[author]

        # If author only posts in one subreddit, no contact
        if len(subreddits) < 1:
            continue

        # Generate all possible pairs
        pairs = permutations(subreddits, 2)

        # Increment overlap for each pair
        for a, b in pairs:
            x = ROWS.index(a)
            y = ROWS.index(b)
            user_overlap[x,y] += 1
            
    return user_overlap




def size_wrapper(activity_dict):
    # Compute size
    print("Computing size value...")
    subreddit_size_dict = compute_subreddit_size(activity_dict)
    return subreddit_size_dict


def loyalty_wrapper(activity_dict):
    # Compute loyalty
    print("Computing loyalty...")
    author_top_level_posting_activity = compute_author_activity(activity_dict, 10)
    author_loyalty_dict, subreddit_loyalty_dict = compute_subreddit_loyalty(activity_dict, author_top_level_posting_activity, 0.50)
    return subreddit_loyalty_dict



def density_wrapper(activity_dict, load_from_scratch):

    print("Computing density...")
    author_activity_dict = compute_author_activity(activity_dict, 0)
    top_active_authors = extract_top_active_authors(author_activity_dict, 0.2)
    
    if load_from_scratch:
        print("From scratch...")
        pool_data = [{"filename": file, "active_authors": top_active_authors, "relevant_communities": attested_communities_set} for file in all_files]

        with Pool(len(all_files)) as p:
            r = list(tqdm(p.imap(generate_subreddit_author_metadata, pool_data), total=len(pool_data)))
        for sap_dict, filename in r:
            write_dict_to_json(sap_dict, f"{DICTIONARY_DIR}/{filename}_author_to_parent_ids.json")
    
        del r
        gc.collect()

        subreddit_to_author_to_parent_ids, id_to_author = load_density_metadata(all_files, top_active_authors)
        generate_density_edges(subreddit_to_author_to_parent_ids, id_to_author)
        
    subreddit_density_dict = compute_density(attested_communities_set)
    return subreddit_density_dict


def contact_wrapper(activity_dict):
    # Constants
    UNIGRAMS_DIRECTORY = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/network_analysis/word_based_textual_representations/"
    UNI_SVD_TFIDF = UNIGRAMS_DIRECTORY + 'svd-tf-idf_unigrams_sentence_df.npy'
    UNI_ROWS = UNIGRAMS_DIRECTORY + 'unigram_rows_sentence_df'

    # Load index
    with open(UNI_ROWS,'r') as data_file:
        ROWS = [line.strip() for line in data_file]

    print("Computing contact...")

    # Load user overlap matrix
    print("Loading overlap matrix...")
    user_overlap = load_npz(CACHE + "overlap_9388.npz")
    user_overlap = user_overlap.toarray()

    # Load SVD matrix
    print("Loading SVD TF-IDF matrix...")
    SVD = np.load(UNI_SVD_TFIDF)

    # Create similarity matrix
    print("Creating similarity matrix...")
    similarity = cosine_similarity(SVD, SVD) 

    print("Calculating normalized contact...")
    # Contact Calculation:    
    # Denominator: Sum of all overlaps
    print("Denominator")
    denominators = np.sum(user_overlap, axis=1)
    print(denominators.shape)
    # Numerator: Overlap * Similarity
    print("Numerator")
    numerators = np.sum(user_overlap * similarity, axis=1)
    print(numerators.shape)
    # Contact: Numerator/Denominator
    print("Division")
    contact = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators!=0)
    print(contact.shape)
  
    print("Contructing dictionary...")
    # Construct dictionary to return
    ret = {}
    for i, row in enumerate(ROWS):
        ret[row] = contact[i]
    
    return ret


def main(features, load_from_cache, load_data_from_scratch, load_density_from_scratch):

    feature_to_function = {
        "size": lambda x: size_wrapper(x),
        "density": lambda x: density_wrapper(x, load_from_scratch=load_density_from_scratch),
        "loyalty": lambda x: loyalty_wrapper(x),
        "contact": lambda x: contact_wrapper(x)
    }


    if load_data_from_scratch:
        with Pool(len(all_files)) as p:
            r = list(tqdm(p.imap(generate_mapping, all_files), total=len(all_files)))

        for all_comments_dict, top_level_dict, filename in r:
            write_dict_to_json(all_comments_dict, f"{DICTIONARY_DIR}/{filename}.json")
            write_dict_to_json(top_level_dict, f"{DICTIONARY_DIR}/{filename}_top.json")
        
        del r
        gc.collect()

        for file in all_files:
            generate_density_id_metadata(file)
    
    if load_from_cache:
        print("Loading from Cache...")
        with open(CACHE + "condensed_binary.json", "rb") as f:
            total = orjson.loads(f.read())
    else:        
        total = condense(all_files, top_level_only=False)
        total_top_level = condense(all_files, top_level_only=True)

    feature_dfs = []
    for feature in features:
        feature_func = feature_to_function[feature]
        if feature == 'loyalty':
            out_dict = feature_func(total_top_level)
        else:
            out_dict = feature_func(total)
        df = pd.DataFrame.from_dict(out_dict, orient='index', columns=[feature])
        feature_dfs.append(df)
    
    full_df = pd.concat(feature_dfs, axis=1)
    try:
        relevant_df = full_df.loc[attested_communities_set]
    except:
        attested_communities = attested_communities.difference({"AskReddit"})
        relevant_df = full_df.loc[attested_communities_set]
    relevant_df.to_csv("/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/network_analysis/cogsci_2024_social_network_features.csv")


if __name__ == "__main__":
    # Do not use load from cache when computing loyalty
    features = ['contact']
    main(features, load_from_cache=True, load_data_from_scratch=False, load_density_from_scratch=False)