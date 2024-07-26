
from utils.core_utils import *
from utils.data_utils import load_intensifiers, extract_relevant_markers
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool


# Set up constants
YEARS = np.arange(2011, 2021)
NUM_CORES = 6
months =  ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

model_name="bert-large-nli-mean-tokens"
model = SentenceTransformer(model_name)
rel_markers = load_intensifiers("../data/luo_intensifiers.txt")


def get_sbert_length(sens):
    """Get the length of a sentence in terms of tokens used
    in SBERT's default tokenizer."""
    tokens = model.tokenize(sens)
    lens = [len(token_set) for token_set in tokens]
    return lens


def subreddit_marker_usage_per_month(filename, in_dir, out_dir, prefix):
    #TODO: counts from this function do not match up to Luo Intensifiers Frequency
    subs = defaultdict(Counter)
    batch_size = 1000000
    batch_count = 1
    FIELDS_TO_KEEP = ['author', 'subreddit', "created_utc", 'id', 'body']

    print(filename)

    with open(f"{in_dir}{filename}", "r") as f:
        with open(f"{out_dir}{filename}", "w") as fout:
            curr_bodies = []
            curr_lines = []
            counter = 0
            for line in tqdm(f):
                fline = json.loads(line)
                for line in sent_tokenize(fline["body"]):
                    line_markers = extract_relevant_markers(line, rel_markers)
                    if len(line_markers) != 1: # Remove sentences that have more than one intensifier
                        continue
                    fline = {field: fline[field] for field in FIELDS_TO_KEEP}
                    
                    fline['body'] = line
                    fline['rel_marker'] = line_markers[0]

                    curr_bodies.append(line)
                    curr_lines.append(fline)
                    counter += 1

                if counter >= batch_size:
                    print(f"Batch count: {batch_count}")
                    sentence_lens = get_sbert_length(curr_bodies)
                    for i, curr_len in enumerate(sentence_lens):
                        if curr_len >= 6:
                            out_line = curr_lines[i]
                            fout.write(json.dumps(out_line))
                            fout.write("\n")
                            subs[out_line["subreddit"]][out_line['rel_marker']] += 1
                    curr_lines = []
                    curr_bodies = []
                    counter = 0
                    batch_count += 1

    Serialization.save_obj(subs, f"{prefix}_all_luo_marker_counts_{filename[3:-5]}")


def compute_aggregate_marker_counts(year, prefix):
    files = [f"RC_{year}-{month}.json" for month in months]
    all_counters = []
    for filename in files:
        all_counters.append(Serialization.load_obj(f"{prefix}_all_luo_marker_counts_{filename[3:-5]}"))
    final_counter = reduce((lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) & set(y)}), all_counters)
    Serialization.save_obj(final_counter, f"{prefix}_all_luo_markers_{year}")

def load_subreddit_marker_data(year, prefix):
    subreddit_to_marker_counter = Serialization.load_obj(f"{prefix}_all_luo_markers_{year}")
    df = pd.DataFrame(subreddit_to_marker_counter).fillna(0).T
    print(df.sum().sort_values().sum())
    return df

def compute_cutoffs(df, cutoff):
    totals = df.sum(axis=1)
    return totals[totals >= cutoff]

def graph_coms_with_markers(df, year):
    cutoffs = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    com_counts = []
    for cutoff in cutoffs:
        num_communities = compute_cutoffs(df, cutoff).shape[0]
        com_counts.append(num_communities)

    plt.plot(np.log10(cutoffs), com_counts, marker='o')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], labels = ["1", "10", "100", "1K", "10K", "100K", "1M"])
    plt.xlabel("Number of comments with stance markers (log-scale)")
    plt.ylabel("Number of communities")
    plt.title(f"Number of communities with at least n \nstance marker posts in {year}")
    plt.show()
    plt.clf()
    plt.plot(np.log10(cutoffs[-5:]), com_counts[-5:], marker='o')
    plt.xticks([4, 5, 6], labels = ["10K", "100K", "1M"])
    plt.xlabel("Number of comments with stance markers")
    plt.ylabel("Number of communities")
    plt.title(f"Number of communities with at least n \nstance marker posts {year} (zoomed in)")
    plt.show()
    plt.clf()


def plot_distinct_markers_per_community(df, year):
    community_totals = df.sum(axis=1)
    freq_coms = community_totals[community_totals>=10000].index
    print(df.shape)
    df = df.loc[freq_coms]
    print(df.shape)
    plt.hist(df.mask(df>0, 1).sum(axis=1))
    plt.xlabel("Number of Distinct Stancemarkers")
    plt.ylabel("Number of Communities")
    plt.title(f"Number of Communities with n Distinct Stancemarkers ({year})")
    plt.show()
    plt.clf()


def extract_communities_in_year(year, prefix):
    """
    Extract all communities with enough sentences with intensifiers
    in the relevant year.
    """
    year_df = load_subreddit_marker_data(year, prefix)
    year_10k_cutoff = compute_cutoffs(year_df, 10000)
    frequent_data = year_df[year_df.index.isin(year_10k_cutoff.index)].T
    marker_distribution_per_community = frequent_data.sort_index()
    marker_distribution_per_community /= marker_distribution_per_community.sum()

    # Save overall marker statistics for sampling 10K sentences
    Serialization.save_obj(marker_distribution_per_community, f"{prefix}_full_data_intensifier_distribution_per_community")

    plt.hist(np.log10(year_10k_cutoff))
    plt.xlabel("# Comments with Intensifiers (Log)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Comment Volumes across Communities (n={len(year_10k_cutoff)})")
    plt.savefig(IMAGES_DIR + f"community_comment_volume_{year}.png")
 

if __name__ == "__main__":

    year = "2018"
    prefix = f"temporal_analysis_{year}"
    IN_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/luo_data/raw_data/"
    OUT_DIR = f"/ais/hal9000/datasets/reddit/stance_pipeline/{prefix}/processed_data/"

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    ### Block 1: For extracting marker usage for each subreddit per month
    all_files = [f"RC_{year}-{month}.json" for month in months]
    for file in all_files:
        subreddit_marker_usage_per_month(file, IN_DIR, OUT_DIR, prefix)
    # with Pool(NUM_CORES) as p:
    #     r = list(tqdm(p.imap(subreddit_marker_usage_per_month, all_files), total=len(all_files)))

    # ### Block 1: End

    ### Block 2: For aggregating marker counts across the Serialized monthly objects
    compute_aggregate_marker_counts(year, prefix)
    extract_communities_in_year(year, prefix)
    