from utils.core_utils import *
from utils.data_utils import extract_relevant_markers, load_intensifiers
from nltk import sent_tokenize
import re
import gc
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool

model_name="bert-large-nli-mean-tokens"
model = SentenceTransformer(model_name)
rel_markers = load_intensifiers("../data/luo_intensifiers.txt")

def get_sbert_length(sens):
    """Get the length of a sentence in terms of tokens used
    in SBERT's default tokenizer."""
    tokens = model.tokenize(sens)
    lens = [len(token_set) for token_set in tokens]
    return lens



def filter_df(df):
    """
    Apply inclusion criteria to current dataset.
    """
    tmp = df[['author', 'subreddit', "created_utc", 'id', 'body']].reset_index(drop=True)
    tmp['rel_marker'] = tmp['body'].apply(lambda x: extract_relevant_markers(x, rel_markers))
    tmp['one_marker'] = tmp['rel_marker'].apply(lambda x: len(x) == 1)
    tmp = tmp[tmp['one_marker']]
    tmp['rel_marker'] = tmp['rel_marker'].apply(lambda x: x[0])
    tmp['len'] = get_sbert_length(tmp['body'].tolist())
    return tmp[tmp['len'] >= 6]



def extract_test_data(test_communities_file, data_dir, sub_file, output_dir):

    # To avoid recreating files, we skip all data that has already been created
    print(data_dir)
    # if os.path.exists(f"{output_dir}/{sub_file[:-5]}.csv"):
    #     return
    
    rel_communities = pd.read_csv(test_communities_file, index_col=0, header=None).index.tolist()
    print(f"Number of relevant communities: {len(rel_communities)}")

    df = pd.read_json(data_dir + "/" + sub_file, lines=True)
    df = df[df['subreddit'].isin(rel_communities)]
    df['body_mask'] = df.apply(lambda x: re.sub(x['rel_marker'], "[MASK]", x['body'].lower()), axis=1)
    df.to_csv(f"{output_dir}/{sub_file[:-5]}.csv", index=False)

    del df
    gc.collect()

    # agg = pd.concat(curr_total)
    # # Mask sentences
    # agg['body_mask'] = agg.apply(lambda x: re.sub(x['rel_marker'][0], "[MASK]", x['body'].lower()), axis=1)
    # agg.to_csv(f"{output_dir}/{data_dir[idx + 1: ]}.csv")


def sample_community_data(data_dict):
    sub = data_dict['df']
    marker_distribution = data_dict['marker_dist']
    com = sub.iloc[0]['subreddit']
    # We want all AskReddit data
    if com == "AskReddit":
        return sub['id'].tolist()
    else:
        big_sample = sub.groupby("rel_marker").apply(lambda x: x.sample(int(10000 * marker_distribution[com].loc[x.name]) + 1, random_state=42)).reset_index(drop=True)
        curr_sample = big_sample.sample(10000, random_state=42)
        return curr_sample['id'].tolist()


def sample_sentences(data_dir, output_dir, prefix):
    # Load marker distribution
    marker_distribution = Serialization.load_obj(f"{prefix}_full_data_intensifier_distribution_per_community")

    # Load all of the CSVs into memory (only need the subreddit, id, and rel_marker columns)
    print("Loading data into memory...")
    dfs = []
    for file in tqdm(os.listdir(data_dir)):
        df = pd.read_csv(data_dir + file, usecols=['subreddit', 'id', 'rel_marker'])
        dfs.append(df)
    all_dfs = pd.concat(dfs)
    print(all_dfs.shape)
    print(all_dfs['subreddit'].nunique())

    all_dfs['sen_id'] = all_dfs.groupby("id").cumcount()
    all_dfs['id'] = all_dfs['id'] + "-" +  all_dfs['sen_id'].astype(str)
    all_dfs = all_dfs[['subreddit', 'id', 'rel_marker']]

    # Now loop through each subreddit and sample 10K
    print("Sampling...")
    sampled_ids = []
    com_dfs = []
    for community in tqdm(marker_distribution.columns):
        sub = all_dfs[all_dfs['subreddit'] == community]
        com_dfs.append({"df": sub, "marker_dist": marker_distribution})
    
    with Pool(12) as p:
        r = list(tqdm(p.imap(sample_community_data, com_dfs), total=len(com_dfs)))

    sampled_ids = flatten(r)
    print(f"Number of sampled ids: {len(sampled_ids)}")

    #TODO: REDO ANALYSIS BUT GET RID OF DUPLICATE IDS, WE SHOULD HAVE SAMPLED SENTENCES NOT IDS

    print("Saving sampled ids...")
    Serialization.save_obj(sampled_ids, "cogsci_2024_sampled_comment_ids")

    print("Saving sampled data...")
    for file in tqdm(os.listdir(data_dir)):
        df = pd.read_csv(data_dir + file)
        df['sen_id'] = df.groupby("id").cumcount()
        df['id'] = df['id'] + "-" +  df['sen_id'].astype(str)
        df = df[['subreddit', 'id', 'rel_marker']]
        df = df[df['id'].isin(sampled_ids)]
        df.to_csv(output_dir + file, index=False)


    # com_to_df = {}
    # for community in tqdm(marker_distribution.columns):

    #     sub = all_dfs[all_dfs['subreddit'].str.lower() == community.lower()]
    #     com_to_df[community] = sub

    #     # We want all AskReddit data

    #     if community == "AskReddit":
    #         sampled_ids.extend(sub['id'].tolist())
    #     else:
    #         big_sample = sub.groupby("rel_marker").apply(lambda x: x.sample(int(10000 * marker_distribution[community].loc[x.name]) + 1, random_state=42)).reset_index(drop=True)
    #         print(big_sample.shape)
    #         curr_sample = big_sample.sample(10000, random_state=42)
    #         sampled_ids.extend(curr_sample['id'].tolist())

    # print("Saving sampled ids...")
    # Serialization.save_obj(sampled_ids, "cogsci_2024_sampled_comment_ids")

    # for file in tqdm(os.listdir(data_dir)):
    #     df = pd.read_csv(data_dir + file)
    #     df = df[df['id'].isin(sampled_ids)]
    #     df.to_csv(output_dir + file, index=False)



def mask_sentences(directory):
    files = sorted(os.listdir(directory))
    print(files)
    for file in files:
        df = pd.read_csv(directory + file, index_col=0).reset_index(drop=True)
        df['body_mask'] = df.progress_apply(lambda x: re.sub(x['rel_marker'], "[MASK]", x['body'].lower()), axis=1)
        df.to_csv(directory + file)



# def extract_intensifier_sentences():
#     for dir_tup in files:
#         dir = dir_tup[0]
#         if not ("2019" in dir):
#             continue
#         sub_files = sorted(dir_tup[2])
#         print(dir)
#         curr_total = []
#         for sub_file in tqdm(sub_files):
#             df = pd.read_json(dir + "/" + sub_file, lines=True)
#             tmp = df[['author', 'body', 'subreddit', 'id', "created_utc", "BF_markers"]]
#             tmp['subreddit'] = tmp['subreddit'].str.lower()
#             curr_total.append(filter_df(tmp))
#         idx = data_dir.rfind("/")
#         agg = pd.concat(curr_total)
#         # Mask sentences
#         agg['body_mask'] = agg.apply(lambda x: re.sub(x['rel_marker'][0], "[MASK]", x['body'].lower()), axis=1)
#         agg.to_csv(f"{output_dir}/{data_dir[idx + 1: ]}.csv")


# if __name__ == '__main__':
    


########### POSTING DISTRIBUTION FOR NEW SENTENCES STARTS HERE


    # intensifiers = load_intensifiers()
    # ROOT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/"
    # files = sorted(list(os.walk(ROOT_DIR)))
    # for dir_tup in files:
    #     dir = dir_tup[0]
    #     if not dir.endswith("files"):
    #         continue
    #     dir_prefix = dir[dir.rindex("/") + 1:]
    #     if not dir_prefix.startswith("2014"):
    #         continue
    #     sub_files = sorted(dir_tup[2])
    #     print(dir_prefix[:7])
    #     total = None
    #     for sub_file in tqdm(sub_files):
    #         df = pd.read_json(dir + "/" + sub_file, lines=True)
    #         df = df[df['BF'] == 1]
    #         df['rel_marker'] = df['BF_markers'].apply(lambda x: x.split("__"))
    #         df['len'] = df['rel_marker'].apply(len)
    #         df = df[df['len'] == 1]
    #         df['rel_marker'] = df['rel_marker'].apply(lambda x: x[0])
    #         df = df[df['rel_marker'].isin(intensifiers)]
    #         df = df[['subreddit', 'rel_marker', 'id', 'created_utc']]
    #         agg = df.groupby(["subreddit", "rel_marker"]).nunique()
    #         if total is None:
    #             total = agg
    #         else:
    #             total = total.add(agg, fill_value=0)
        
    #     total.to_csv(f"/u/jai/stancetaking/data_files/community_posting_statistics_{dir_prefix[:7]}.csv")