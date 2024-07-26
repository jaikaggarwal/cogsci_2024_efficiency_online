"""
Manual tf-idf since we only have word counts. 
The misalignment paper takes in consideration
"common" one to four grams but 
it's so time intensive to get ngrams.  
"""

from utils.core_utils import *
from collections import defaultdict, Counter
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool
import time
import math


DATA = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/social_network_data/"

UNIGRAMS_DIRECTORY = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/network_analysis/word_based_textual_representations/"
UNIGRAMS = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/network_analysis/textual_representations/" + "all_unigrams_sentence_df.json"
UNI_TFIDF = UNIGRAMS_DIRECTORY + 'tf-idf_unigrams_sentence_df.npy' 
UNI_SVD_TFIDF = UNIGRAMS_DIRECTORY + 'svd-tf-idf_unigrams_sentence_df.npy'
UNI_VOCAB = UNIGRAMS_DIRECTORY + 'unigram_vocab_sentence_df'
UNI_ROWS = UNIGRAMS_DIRECTORY + 'unigram_rows_sentence_df'

data_suffix = "cogsci_2024"
attested_communities = pd.read_csv("../data/cogsci_2024_communities.txt", header=None)
attested_communities_set = set(attested_communities[0].tolist())


def write_dict_to_json(data, path):
    with open(path,"w") as f:
        json.dump(data, f)


def load_dict_from_json(path):
    with open(path,"r") as f:
        return json.load(f)


def get_batches(arr, num_batches):
    batch_size = len(arr) // num_batches
    batches = []
    for i in range(num_batches+1):
        batches.append(arr[i*batch_size: (i+1)*batch_size])
    
    # assert len(flatten(batches)) == len(arr)
    return batches


def compute_unigram_dict(batch):
    sub_unigrams = defaultdict(Counter)
    for sub, line in batch:
        sub_unigrams[sub] += Counter(map(lambda x: x.lower(), line))
    return sub_unigrams


def get_unigrams(filename):

    path = DATA + filename + ".json"
    line_count = 0
    subreddit_counter = defaultdict(int)
    all_bodies = []
    with open(path,  "r") as file:
        for line in tqdm(file, total=82000000):
            obj = json.loads(line)
            sub = obj["subreddit"]
            subreddit_counter[sub] += 1

            line_count += 1
            # if line_count >= 1500000:
            #     break
            if subreddit_counter[sub] % 10 != 0:
                continue
            comment = obj['body']
            all_bodies.append((sub, comment))


    
    print("Computing batches...")
    batches = get_batches(all_bodies, 10)

    print("Multiprocessing...")
    with Pool(len(batches)) as p:
        r = list(tqdm(p.imap(compute_unigram_dict, batches), total=len(batches)))

    print("Merging pools...")
    all_data = defaultdict(Counter)
    for counter in tqdm(r):
        for sub in counter:
            all_data[sub] += counter[sub]

    write_dict_to_json(all_data, f"{UNIGRAMS_DIRECTORY}/{filename}_unigrams.json")




# def get_unigrams_wrapper(dfs):
    
#     # files = glob.glob(folder + "/*")
#     with Pool(4) as p:
#         r = list(tqdm(p.imap(get_unigrams, dfs), total=len(dfs)))
    
#     z = defaultdict(Counter)
#     for sub_unigrams in r:
#         for sub in sub_unigrams:
#             z[sub] += sub_unigrams[sub]
    
#     # # folder_name = folder.split("/")[-2]
#     # with open(UNIGRAMS_DIRECTORY + , "w") as outfile:
#     #     json.dump(z, outfile)


def combine_unigrams(filenames):
    try:
        z = load_dict_from_json(UNIGRAMS)
    except:
        file_paths = [f"{UNIGRAMS_DIRECTORY}/{filename}_unigrams.json" for filename in filenames]
        z = defaultdict(Counter)
        for file_path in tqdm(file_paths):
            unigram_data = load_dict_from_json(file_path)
            for sub_counter in unigram_data:
                z[sub_counter] += unigram_data[sub_counter]
        write_dict_to_json(z, UNIGRAMS)
    return z
    

def get_idf(unigrams): 
    '''
    For each word, get log_10(N/df) where N is 
    number of subreddits. 
    Set min_df to be 5 and max_df to be 0.95
    to filter out too rare and too common words.
    '''
 
    dfs = Counter()    
    for subreddit in unigrams:
        words = unigrams[subreddit].keys()
        words = set(words)
        
        for w in words:
            dfs[w] += 1
    
    idfs = {}
    num_sr = len(unigrams)
    print(num_sr)
    
    for w in tqdm(dfs):
        if w.isalpha():
            if dfs[w] > 5 and dfs[w] <= 0.95 * num_sr:
                idfs[w] = math.log10(num_sr / float(dfs[w]))
    return idfs
    

def get_tf_idf(tfs, idfs):
    '''
    For each word in idfs, get its (1 + log tf) 
    for each document. 
    Create vectors for each document where each 
    index corresponds to a word and the value is
    (1 + log tf)xlog_10(N/df). 
    '''
    
    VOCAB = UNI_VOCAB
    ROWS = UNI_ROWS
    
    vocab = sorted(idfs.keys())
    srs = []
    X = []
    num_sr = 0

    for sr in tqdm(tfs.keys()): 
        srs.append(sr)
        vec = np.zeros(len(vocab))
        for i, w in enumerate(vocab): 
            if tfs[sr].get(w, 0) > 0: 
                vec[i] = (1 + math.log10(tfs[sr][w]))*idfs[w]
        X.append(vec)
    X = np.array(X)
    print(X.shape)
    # np.save(OUTPUT, X)
    with open(ROWS, 'w') as outputfile: 
        for sr in srs: 
            outputfile.write(sr + '\n') 
    with open(VOCAB, 'w') as outputfile: 
        for w in vocab: 
            outputfile.write(w + '\n') 
    
    # print(1 - spatial.distance.cosine(X[srs.index('android')], X[srs.index('apple')]))
    # print(1 - spatial.distance.cosine(X[srs.index('london')], X[srs.index('ukpolitics')]))
    # print(1 - spatial.distance.cosine(X[srs.index('london')], X[srs.index('android')]))

    return X



    
def svd_tf_idf(X): 
    INPUT = UNI_TFIDF
    OUTPUT = UNI_SVD_TFIDF
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    print("Fitting SVD...")
    X_new = svd.fit_transform(X)
    Serialization.save_obj(svd, "cogsci_2024_textual_tf_idf_svd_sentence_df")
    print("Normalizing...")
    normalizer = Normalizer(copy=False)
    X_new_new = normalizer.fit_transform(X_new)
    print("Saving")
    np.save(OUTPUT, X_new_new)
    

def main(): 
    # filenames = os.listdir(UNIGRAMS_DIRECTORY)
    # filenames = [UNIGRAMS_DIRECTORY + file for file in filenames if file.endswith("json")]
    # print(filenames)
    filenames = ["/ais/hal9000/datasets/reddit/brian_unigrams/sentence_df_unigrams.json"]
    print("Computing tfs...")
    tfs = combine_unigrams(filenames)
    print("Computing idfs...")
    idfs = get_idf(tfs, 'text') 
    print("Computing tf-idf...")
    tf_idf = get_tf_idf(tfs, idfs, 'text')
    print("Computing SVD...")
    svd_tf_idf(tf_idf, 'text')



if __name__ == "__main__":
    file_prefix = "RC_2019-"
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    all_files = [file_prefix + month for month in months]
    print(all_files)

    # for file in all_files:
    #     a = time.time()
    #     get_unigrams(file)
    #     b = time.time()
    #     print(f"Total time for {file}: {(b-a)/60} minutes")
    print("Computing tfs...")
    tfs = combine_unigrams(all_files)
    print("Computing idfs...")
    idfs = get_idf(tfs) 
    print("Computing tf-idf...")
    tf_idf = get_tf_idf(tfs, idfs)
    print("Computing SVD...")
    svd_tf_idf(tf_idf)
