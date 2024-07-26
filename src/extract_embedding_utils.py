
from utils.core_utils import *
from sentence_transformers import SentenceTransformer
import re

from sentence_transformers.SentenceTransformer import torch as pt
pt.cuda.set_device(0)


class SBERT:
    """Wrapper class for SBERT, used to encode text."""
    def __init__(self, model_name="bert-large-nli-mean-tokens") -> None:
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, data):
        pt.cuda.empty_cache()
        embeddings = self.model.encode(data, show_progress_bar=True)
        return embeddings

    def get_tokenize_lengths(self, data, remove_special_tokens=True):
        """Get number of token segments in the SBERT tokenization of sentences.
        
        Optionally include or exclude the special <CLS> and <SEP> tokens that are
        added automatically.
        """
        # If this is just a raw string, nest it in a list
        if type(data) == str:
            data = [data]
        tokenizations = self.model.tokenize(data)["input_ids"]

        # Since padding tokens are tokenized with id 0, we just count
        # non zero elements.

        # If we have more than 1 post, vectorize counting over each post.
        if tokenizations.dim() > 1:
            counts = tokenizations.count_nonzero(axis=1)
        else:
            counts = tokenizations.count_nonzero()

        if remove_special_tokens:
            # Remove the 2 special tokens
            counts = counts - 2
        return counts


def process_datadumps(dump_file, sbert, post_level_embeds_output_dir, metadata_output_dir, to_mask, swap_mask_for_blank):
    """Process Reddit data dumps csv into metadata and SBert embeddings.
    
    The embeddings are separated into post-level embeddings where each post
    gets a singular embedding as calculated by one encoding of SBERT and
    sentence-level embeddings where each sentence in a post gets its own
    separate embedding."""

    data = pd.read_csv(dump_file)
    if swap_mask_for_blank:
        data['body_mask'] = data['body_mask'].progress_apply(lambda x: re.sub("\[MASK\]", "", x))

    # This is now taken care of in data_processing.py
    # data['sen_id'] = data.groupby("id").cumcount()
    # data['id'] = data['id'] + "-" +  data['sen_id'].astype(str)

    # Create post-level embeddings
    NUM_SPLITS = 40
    data_splits = np.array_split(data, NUM_SPLITS)
    for i, split in enumerate(data_splits):

        if i >= 10:
            output_file_name = f"_0{i}.csv"
        else:
            output_file_name = f"_00{i}.csv"
        if os.path.exists(post_level_embeds_output_dir + output_file_name):
            continue
        split = split.reset_index(drop=True)
        if to_mask:
            all_text_embeddings = sbert.get_embeddings(split.body_mask.tolist())
        else:
            all_text_embeddings = sbert.get_embeddings(split.body.tolist())
        all_text_df = pd.DataFrame(all_text_embeddings)
        all_text_df = pd.concat([split.id, all_text_df], axis=1)
        # all_text_dfs.append(all_text_df)
        all_text_df.to_csv(post_level_embeds_output_dir + output_file_name, index=False)     

    # Create metadata csv
    # post_lengths = pd.Series(post_lengths, name="sentence_count")
    metadata = data[["id", "subreddit", "body", "created_utc", "rel_marker", "body_mask"]]
    # metadata = pd.concat([metadata, post_lengths], axis=1)
    metadata.to_csv(metadata_output_dir, index=False)

def embeddings_wrapper(input_dir, output_dir, sbert_model, range_start, range_finish, to_mask, swap_mask_for_blank):
    print(input_dir)
    print(os.listdir(input_dir))
    files = sorted(os.listdir(input_dir))[range_start:range_finish]
    print(files)
    for file in tqdm(files):
        print(file)
        process_datadumps(
            input_dir + file, 
            sbert_model, 
            output_dir + file[:-4] + "_embeddings", 
            output_dir + file[:-4] + "_metadata.csv",
            to_mask,
            swap_mask_for_blank
        )

if __name__ == "__main__":
    pass

    # ### Block 1: Use the block below to run this code on a single file
    # INPUT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/test_run_data/"
    # OUTPUT_DIR = "/ais/hal9000/datasets/reddit/jai_stance_embeddings/"
    # sbert_model = SBERT("bert-large-nli-mean-tokens")
    # files = os.listdir(INPUT_DIR)
    # for file in tqdm(files):
    #     print(file)
    #     process_datadumps(INPUT_DIR + file, sbert_model, OUTPUT_DIR + file[:-4] + "_embeddings.csv", OUTPUT_DIR  + file[:-4] +  "_metadata.csv")