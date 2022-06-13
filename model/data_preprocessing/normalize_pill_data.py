import cv2
import os
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt


def subimage(image, rect):
    theta = rect[2] - 90
    center = (int(rect[0][0]), int(rect[0][1]))
    height = int(rect[1][0])
    width = int(rect[1][1])
    theta *= 3.14159 / 180  # convert to rad
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((width - 1) / 2) - v_y[0] * ((height - 1) / 2)
    s_y = center[1] - v_x[1] * ((width - 1) / 2) - v_y[1] * ((height - 1) / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(
        image,
        mapping,
        (width, height),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize(args):
    if args.excel_file_name.split(".")[-1] == "xls":
        df = pd.read_excel(args.excel_file_name, engine="openpyxl")
    elif args.excel_file_name.split(".")[-1] == "csv":
        df = pd.read_csv(args.excel_file_name)

    for i, row in tqdm(df.iterrows()):
        index = row["품목일련번호"]

        img = cv2.imread(
            os.path.join(args.background_removed_data_dir, str(index) + ".png")
        )
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        pills = []
        counts = 0
        inds = []
        for j, contr in enumerate(contours):
            rect = cv2.boundingRect(contr)
            if rect[3] * rect[2] > 1000:
                counts += 1
                inds.append(j)

        if len(inds) == 2:
            for ind in inds:
                if row["의약품제형"] == "원형":
                    rect = cv2.boundingRect(contours[ind])
                    rect = (
                        (rect[0] + rect[3] // 2, rect[1] + rect[2] // 2),
                        (rect[3], rect[2]),
                        90,
                    )
                else:
                    rect = cv2.minAreaRect(contr)
                crop = subimage(img, rect)
                if crop.shape[0] > crop.shape[1]:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                crop = cv2.resize(crop, (256, int(256 * crop.shape[0] / crop.shape[1])))
                pills.append(crop)
            normalized = np.zeros((512, 512, 3), dtype=np.uint8)
            normalized[
                128
                - pills[0].shape[0] // 2 : 128
                + (pills[0].shape[0] - pills[0].shape[0] // 2),
                128
                - pills[0].shape[1] // 2 : 128
                + (pills[0].shape[1] - pills[0].shape[1] // 2),
            ] = pills[0]
            normalized[
                384
                - pills[1].shape[0] // 2 : 384
                + (pills[1].shape[0] - pills[1].shape[0] // 2),
                384
                - pills[1].shape[1] // 2 : 384
                + (pills[1].shape[1] - pills[1].shape[1] // 2),
            ] = pills[1]
            cv2.imwrite(
                os.path.join(args.success_data_save_dir, str(index) + ".png"),
                normalized,
            )
        else:
            print(i)
            cv2.imwrite(os.path.join(args.fail_data_save_dir, str(index) + ".png"), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel_file_name",
        type=str,
        default="../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv",
        help="path of the xls/csv file (default: ../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv)",
    )
    parser.add_argument(
        "--background_removed_data_dir",
        type=str,
        default="../data/background_removed_data",
        help="path of the xls/csv file (default: ../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv)",
    )
    parser.add_argument(
        "--success_data_save_dir",
        type=str,
        default="../data/normalized_data",
        help="path of the xls/csv file (default: ../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv)",
    )
    parser.add_argument(
        "--fail_data_save_dir",
        type=str,
        default="../data/normalized_failed_data",
        help="path of the xls/csv file (default: ../pill_excel_data/OpenData_PotOpenTabletIdntfcC20220601.csv)",
    )

    args = parser.parse_args()
    normalize(args)
