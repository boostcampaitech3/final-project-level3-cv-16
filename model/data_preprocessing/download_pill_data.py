"""
https://nedrug.mfds.go.kr/pbp/CCBGA01/getItem?totalPages=4&limit=10&page=2&&openDataInfoSeq=11
must download the excel file from the link above
"""

import time
import argparse
import pandas as pd
import urllib.request as req
from tqdm import tqdm


def download(args):
    if args.excel_file_name.split(".")[-1] == "xls":
        df = pd.read_excel(args.excel_file_name, engine="openpyxl")
    elif args.excel_file_name.split(".")[-1] == "csv":
        df = pd.read_csv(args.excel_file_name)

    start = time.time()
    for idx in tqdm(range(len(df))):
        image_key = list(df["품목일련번호"])[idx]
        image_url = list(df["큰제품이미지"])[idx]
        req.urlretrieve(image_url, f"{args.data_save_dir}/{image_key}.jpg")
    print(time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel_file_name",
        type=str,
        default="../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv",
        help="path of the xls/csv file (default: ../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv)",
    )
    parser.add_argument(
        "--data_save_dir",
        type=str,
        default="../data/raw_data",
        help="path to saving downloaded images (default: ../data/raw_data)",
    )

    args = parser.parse_args()
    download(args)
