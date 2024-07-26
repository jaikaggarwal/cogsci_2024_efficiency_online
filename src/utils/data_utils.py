
from .core_utils import *
import re
import time
import zstandard as zstd
from nltk import word_tokenize
# from sentence_transformers import SentenceTransformer
import string

# model_name="bert-large-nli-mean-tokens"
# model = SentenceTransformer(model_name)


FEATURE_COLUMNS = ['Valence', 'Arousal', 'Dominance', 'Politeness', 'Formality']
NUM_QUANTILES = 4
ROOT_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/luo_tiny_test/full_features/"

translator = str.maketrans('', '', string.punctuation) 

def load_intensifiers(txt_list):
    """
    Loads the set of intensifiers from a text file.
    """

    with open(f"../data/{txt_list}", "r") as intensifiers:
        lines = intensifiers.read().splitlines()
    
    return lines


def extract_relevant_markers(new_line, terms):
    """Collect all of the relevant markers that are present
    within new_line."""
    curr_body = new_line.lower().split(" ")
    present_markers = Counter(curr_body)
    return [term for term in terms if present_markers.get(term, 0) == 1]


def preprocess_heatmap(cooc, marker, community=None):
    """Preprocess cooc dataframe for per word heatmaps.
    
    Pass in a list of community if we want to filter by certain communities."""

    if community:
        cooc = cooc[cooc["community"].isin(community)]
        
    cooc = cooc[cooc["marker"] == marker]

    cooc = pd.pivot_table(cooc, values="value", index="vad", columns="pf")

    return cooc


HTTP_PATTERN = re.compile(r'[\(\[]?https?:\/\/.*?(\s|\Z)[\r\n]*')
def preprocess_comment(text):
    """
    Preprocesses text from Reddit posts and comments.
    """
    # Replace links with LINK token
    line = HTTP_PATTERN.sub(" LINK ", text)
    # Replace irregular symbol with whitespace
    line = re.sub("&amp;#x200b", " ", line)
    # Replace instances of users quoting previous posts with empty string
    line = re.sub(r"&gt;.*?(\n|\s\s|\Z)", " ", line)
    line = line.translate(translator)
    # # Replace extraneous parentheses with whitespace
    # line = re.sub(r'\s\(\s', " ", line)
    # line = re.sub(r'\s\)\s', " ", line)
    # # Replace newlines with whitespace
    # line = re.sub(r"\r", " ", line)
    # line = re.sub(r"\n", " ", line)
    # # Replace mentions of users with USER tokens
    # line = re.sub("\s/?u/[a-zA-Z0-9-_]*(\s|\Z)", " USER ", line)
    # # Replace mentions of subreddits with REDDIT tokens
    # line = re.sub("\s/?r/[a-zA-Z0-9-_]*(\s|\Z)", " REDDIT ", line)
    # # Replace malformed quotation marks and apostrophes
    # line = re.sub("’", "'", line)
    # line = re.sub("”", '"', line)
    # line = re.sub("“", '"', line)
    # Get rid of asterisks indicating bolded or italicized comments
    # line = re.sub("\*{1,}(.+?)\*{1,}", r"\1", line)    
    # Replace emojis with EMOJI token
    # line = emoji.get_emoji_regexp().sub(" EMOJI ", line)
    # Replace all multi-whitespace characters with a single space.
    # line = re.sub("\s{2,}", " ", line)
    return line


def contentful_filter(new_line):
    """Filter out comments that do not have meaningful content."""
    if new_line['body'] is np.nan:
        return False
    if new_line['body'].lower() in ['[deleted]', '[removed]']:
        return False
    # Remove all bots, moderators, deleted, removed authors (and spammer dollarwolf)
    if new_line['author'] in ["AutoModerator", "dollarwolf", "[deleted]", "[removed]"]:
        return False
    elif new_line['author'].lower().endswith("bot"):
        return False
    return True


def apply_filters(new_line, length_filter='word_tokenize'):
    # Now we can check to see if the body is long enough
    # if to_keep:
    new_body = preprocess_comment(new_line['body'])
        # FIX THIS
    if length_filter == 'sbert':
        new_body = model.tokenize(new_body)
        if len(new_body) < 6:
            return False, new_line
    else:
        new_body = word_tokenize(new_body)
        if len(new_body) < 6:
            return False, new_line

    new_line['body'] = new_body
    return True, new_line
    

def apply_filters_with_term(new_line, terms):
    """"""
   
    curr_body = set(new_line['body'].lower().split(" "))
    any_present = 0
    for key in terms:
        present_markers = [val for val in terms[key] if val in curr_body]
        if len(present_markers) > 0:
            curr_str = "__".join(present_markers)
            new_line[key] = 1
            new_line[key + "_markers"] = curr_str
            any_present += 1
        else:
            new_line[key] = 0
            new_line[key + "_markers"] = ""
    
    if any_present > 0:
        return True, new_line
    else:
        return False, new_line
 
## Extract data
TOP_10K_SUBS = pd.read_csv("../data/reddit-master-metadata.tsv", delimiter="\t")['community'].tolist()
TOP_10K_SUBS = [sub.lower() for sub in TOP_10K_SUBS]

FIELDS_TO_KEEP = ['author', 'body', 'controversiality', 'created_utc', 'id', 'parent_id', 'score', 'subreddit', 'author_flair_text', 'author_flair_css_class']
quarter_to_data = {
    "first": ["01", "02", "03"],
    "second": ["04", "05", "06"],
    "third": ["07", "08", "09"],
    "fourth": ["10", "11", "12"],
}


def group_to_terms(txt_file):
    markers_of_interest = load_intensifiers(txt_file)

    return {
        "BF": markers_of_interest
    }

# Insert custom directories for the data dumps files
DATA_DIR = '/ais/hal9000/datasets/reddit/data_dumps/'

# Directory for output files
OUT_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/luo_data/raw_data/"


def extract_data_with_term(zst_file): #year, quarter
    intensifiers = group_to_terms("luo_intensifiers.txt")
    zst_files = [zst_file] 
    print("Number of files...", len(zst_files))
    for filename in zst_files:
        # Trackers
        print(filename) # Which files
        counter = 0 # How many bytes we've seen
        loops = 0 # How many windows we've decompressed
        a = time.time() # Overall time
        with open(f"{OUT_DIR}{filename[:-4]}.json", "w") as f_out:
            with open(DATA_DIR + filename, 'rb') as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
                with dctx.stream_reader(fh) as reader:
                    previous_line = ""
                    while True:
                        chunk = reader.read(2**24)  # 16mb chunks
                        counter += 2**24
                        loops += 1
                        if loops % 2000 == 0:
                            print(f"{counter/10**9:.2f} GB")
                            print(f"{(time.time()-a)/60:.2f} minutes passed")
                        if not chunk:
                            break

                        string_data = chunk.decode('utf-8')
                        lines = string_data.split("\n")
                        for i, line in enumerate(lines[:-1]):
                            if i == 0:
                                line = previous_line + line
                            line = json.loads(line)

                            if line['subreddit'].lower() not in TOP_10K_SUBS:
                                continue
                            
                            new_line = {field: line.get(field, np.nan) for field in FIELDS_TO_KEEP}
                            to_keep = contentful_filter(new_line)
                            if not to_keep:
                                continue 

                            to_keep, new_line = apply_filters(new_line)
                            if not to_keep:
                                continue

                            f_out.write(json.dumps(new_line))
                            f_out.write("\n")
                            # do something with the object here
                        previous_line = lines[-1]



def extract_social_network_data(filename):
    SOCIAL_NETWORK_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/cogsci_2024/social_network_data/"
    
    # BATCH_SIZE = 500000
    # Trackers
    counter = 0 # How many bytes we've seen
    loops = 0 # How many windows we've decompressed
    # batch_count = 0
    # batch = []
    # line_counter = 0
    a = time.time() # Overall time
    print(filename)
    with open(f"{SOCIAL_NETWORK_DIR}{filename[:-4]}.json", "w") as f_out:
        # batchify this for simpler processing later
        # Add preprocessing only to TF-IDF code (no need now)
        with open(DATA_DIR + filename, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(fh) as reader:
                previous_line = ""
                while True:
                    chunk = reader.read(2**24)  # 16mb chunks
                    counter += 2**24
                    loops += 1
                    if loops % 1000 == 0:
                        print(f"{counter/10**9:.2f} GB")
                        print(f"{(time.time()-a)/60:.2f} minutes passed")
                    if not chunk:
                        break

                    string_data = chunk.decode('utf-8')
                    lines = string_data.split("\n")
                    for i, line in enumerate(lines[:-1]):
                        if i == 0:
                            line = previous_line + line
                        line = json.loads(line)

                        if line['subreddit'].lower() not in TOP_10K_SUBS:
                            continue
                        
                        # line_counter += 1
                        # if line_counter % 10 != 0: # Sample 10% of our data
                        #     continue
                        
                        new_line = {field: line.get(field, np.nan) for field in ['author', 'subreddit', 'id', 'parent_id', 'body']}
                        to_keep = contentful_filter(new_line)
                        if not to_keep:
                            continue 

                        to_keep, new_line = apply_filters(new_line, length_filter="word_tokenize")
                        if not to_keep:
                            continue
                        
                        new_line['is_top_level'] = new_line['parent_id'].startswith("t3")

                        f_out.write(json.dumps(new_line))
                        f_out.write("\n")
                        # batch.append(new_line)
                        # batch_count += 1

                        # if batch_count == BATCH_SIZE:
                        #     for line in batch:
                        #         f_out.write(json.dumps(line))
                        #         f_out.write("\n")

                        #     batch = []
                        #     batch_count = 0

                        # do something with the object here
                    previous_line = lines[-1]



def reducer(accumulator, element):
    """Helper function for merging dictionaries of Counters."""
    for key, value in element.items():
        accumulator[key] += value
    return accumulator


def compute_yearly_aggregate_marker_counts(year):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    files = [f"RC_{year}-{month}.json" for month in months]
    all_counters = []
    
    for filename in files:
        all_counters.append(Serialization.load_obj(f"all_luo_marker_counts_{filename[3:-5]}"))
    
    final_counter = reduce(reducer, all_counters, defaultdict(Counter))
    Serialization.save_obj(final_counter, f"all_luo_markers_{year}")


def normalize_matrix(matrix):
    return matrix/matrix.sum().sum()

def normalize_array(arr):
    return arr/arr.sum()


if __name__ == "__main__":
    save_suffix = "july_23_luo_tiny_test_4_quartiles"
    markers_of_interest = load_intensifiers("../data/luo_intensifiers.txt")
    print(len(markers_of_interest))
    columns_of_interests = FEATURE_COLUMNS
    namespace = "VADPF"
    load_data_from_raw(save_suffix, markers_of_interest, namespace, columns_of_interest=FEATURE_COLUMNS)
    convert_to_cooc_matrix("july_23_luo_tiny_test_4_quartiles")