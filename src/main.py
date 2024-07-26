from utils.core_utils import *
import data_processing as dp
import extract_embedding_utils as eeu
import feature_extraction as fe
import create_joint_distribution as cjd
from multiprocessing import Pool
import re

from sentence_transformers.SentenceTransformer import torch as pt
pt.cuda.set_device(1)


def in_year_range(files, min_year, max_year):
    valid_files = []
    for file in files:
        file_year = int(re.findall("\d{4}", file)[0])
        if min_year <= file_year and file_year <= max_year:
            valid_files.append(file)
    return sorted(valid_files)



def main(config_filename):

    with open(config_filename) as config_file:
        config_data = json.load(config_file)

    # Step 0: Load files and prepare directories
    ROOT_DIR = config_data['root_dir'] + config_data['sub_dir']
    RAW_DATA_DIR = ROOT_DIR + config_data['raw_data_dir']
    PIPELINE_DATA_DIR = ROOT_DIR + config_data['pipeline_data_dir']
    SAMPLE_DIR = ROOT_DIR + config_data['sample_data_dir']
    EMBEDDINGS_DIR = ROOT_DIR + config_data['embeddings_dir']
    SITUATIONS_DIR = ROOT_DIR + config_data['situations_dir']


    range_start = config_data['range_start']
    range_finish = config_data['range_finish']
    num_cores = config_data['num_cores']
    min_year = config_data['min_year']
    max_year = config_data['max_year']
    communities_of_interest_file = config_data['communities_of_interest']

    num_quantiles = config_data["num_quantiles"]
    sem_sit_properties = config_data['semantic_situation_properties']
    save_suffix = config_data['save_suffix']

    
    files = os.listdir(RAW_DATA_DIR)
    files = in_year_range(files, min_year, max_year)

    for dir_str in [ROOT_DIR, PIPELINE_DATA_DIR, SAMPLE_DIR, EMBEDDINGS_DIR, SITUATIONS_DIR]:
        if not os.path.exists(dir_str):
            print(f"Creating directory: {dir_str}")
            os.makedirs(dir_str)
        else:
            print(f"Using existing directory: {dir_str}")
    
    # Step 1: Gather all data from test communities #TODO: Write as wrapper
    # print("Extracting community data...")
    # for file in tqdm(files):
    #     dp.extract_test_data(communities_of_interest_file, RAW_DATA_DIR, file, PIPELINE_DATA_DIR)
    
    # # Step 2: Sample 10K posts per community
    # dp.sample_sentences(PIPELINE_DATA_DIR, SAMPLE_DIR)

    
    # Step 3: Create embeddings and metadata file (cannot parallelize, based on GPUs)
    # print("Loading SBERT model...")
    # sbert_model = eeu.SBERT("bert-large-nli-mean-tokens")
    # print("Create embeddings...")
    # eeu.embeddings_wrapper(SAMPLE_DIR, EMBEDDINGS_DIR, sbert_model, range_start, range_finish, to_mask=True, swap_mask_for_blank=False)
    
    # Step 4: Infer emotional values to create semantic situations
    # print("Extracting features...")
    # fe.extraction_wrapper(EMBEDDINGS_DIR, SITUATIONS_DIR, np.arange(min_year, max_year+1))

    # Step 5: Construct semantic situation X (subreddit_marker) matrix
    cjd.generate_joint_distribution(save_suffix, num_quantiles, sem_sit_properties, SITUATIONS_DIR)

if __name__ == "__main__":
    main(sys.argv[1])
