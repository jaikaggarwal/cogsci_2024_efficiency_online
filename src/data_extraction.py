from utils.core_utils import *
import argparse
from utils.data_utils import extract_data_with_term, extract_social_network_data
from multiprocessing import Pool



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year",
                        type=str)
    parser.add_argument("--num_cores", type=int, help="Number of cores to use.", default=6)

    args = parser.parse_args()
    print(args.year)
    zst_files = [f'RC_{args.year}-{num}.zst' for num in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]]
    with Pool(args.num_cores) as p:
        r = list(tqdm(p.imap(extract_data_with_term, zst_files), total=args.num_cores))
    # for file in tqdm(zst_files):
    #     extract_social_network_data(file)

