import pandas as pd


def excel2df(excel_file_name, delete_pill_num, project_type, custom_label=True):
    if excel_file_name.split(".")[-1] == "xls":
        df = pd.read_excel(excel_file_name, engine="openpyxl")
    elif excel_file_name.split(".")[-1] == "csv":
        df = pd.read_csv(excel_file_name)

    ## delete 구강정 data
    delete_nums = [200605327, 200605328, 200605329, 200605330, 200605331, 200606263]

    ## delete additional data
    delete_nums += delete_pill_num

    if delete_nums != []:
        index_delete = []
        for delete_num in delete_nums:
            index_delete.append(df[df["품목일련번호"] == delete_num].index)

        ## delete data
        for index in index_delete:
            df = df.drop(index)

    if project_type == "의약품제형":
        pill_type = sorted(list(set(df[project_type])))

        pills = []
        for pill in df[project_type]:
            pills.append(pill_type.index(pill))

        df.insert(29, f"{project_type}_to_label", pills)
        num_classes = len(pill_type)

        print("Excel file to dataframe done!")
        return df, pill_type, num_classes

    elif project_type == "색상앞_2가지":
        pill_type = ["not_white", "white"]
        pills = []
        for color in df["색상앞"]:
            if "하양" not in color:
                pills.append(0)
            else:
                pills.append(1)

        df.insert(29, f"{project_type}_to_label", pills)
        num_classes = len(pill_type)

        print("Excel file to dataframe done!")
        return df, pill_type, num_classes

    elif project_type == "색상앞":
        if custom_label:
            if excel_file_name.split(".")[-1] == "xls":
                pill_type = sorted(
                    [
                        "투명",
                        "연두, 투명",
                        "초록, 투명",
                        "회색/하양, 투명",
                        "갈색/빨강/자주/검정",
                        "노랑/주황",
                        "주황, 투명",
                        "하양, 노랑",
                        "분홍, 투명",
                        "갈색, 투명",
                        "남색/보라",
                        "연두/초록/청록",
                        "보라, 투명",
                        "청록, 투명",
                        "분홍",
                        "하양, 파랑",
                        "하양/하양, 갈색",
                        "빨강, 투명",
                        "파랑, 투명",
                        "파랑",
                        "노랑, 투명",
                    ]
                )

                pills = []
                for color in df["색상앞"]:
                    if color == "갈색" or color == "빨강" or color == "자주" or color == "검정":
                        pills.append(0)
                    elif color == "갈색, 투명":
                        pills.append(1)
                    elif color == "남색" or color == "보라":
                        pills.append(2)
                    elif color == "노랑" or color == "주황":
                        pills.append(3)
                    elif color == "노랑, 투명":
                        pills.append(4)
                    elif color == "보라, 투명":
                        pills.append(5)
                    elif color == "분홍":
                        pills.append(6)
                    elif color == "분홍, 투명":
                        pills.append(7)
                    elif color == "빨강, 투명":
                        pills.append(8)
                    elif color == "연두" or color == "초록" or color == "청록":
                        pills.append(9)
                    elif color == "연두, 투명":
                        pills.append(10)
                    elif color == "주황, 투명":
                        pills.append(11)
                    elif color == "청록, 투명":
                        pills.append(12)
                    elif color == "초록, 투명":
                        pills.append(13)
                    elif color == "투명":
                        pills.append(14)
                    elif color == "파랑":
                        pills.append(15)
                    elif color == "파랑, 투명":
                        pills.append(16)
                    elif color == "하양" or color == "하양, 갈색":
                        pills.append(17)
                    elif color == "하양, 노랑":
                        pills.append(18)
                    elif color == "하양, 파랑":
                        pills.append(19)
                    elif color == "회색" or color == "하양, 투명":
                        pills.append(20)

                df.insert(29, f"{project_type}_to_label", pills)
                num_classes = len(pill_type)

                print("Excel file to dataframe done!")
                return df, pill_type, num_classes

            elif excel_file_name.split(".")[-1] == "csv":
                pill_type = sorted(
                    [
                        "투명",
                        "연두|투명",
                        "초록|투명",
                        "회색/하양|투명",
                        "갈색/빨강/자주/검정",
                        "노랑/주황",
                        "주황|투명",
                        "하양|노랑",
                        "분홍|투명",
                        "갈색|투명",
                        "연두/초록/청록",
                        "보라|투명",
                        "청록|투명",
                        "분홍",
                        "하양|파랑",
                        "하양/하양|갈색",
                        "빨강|투명",
                        "남색/보라",
                        "파랑|투명",
                        "파랑",
                        "노랑|투명",
                    ]
                )

                pills = []
                for color in df["색상앞"]:
                    if color == "갈색" or color == "빨강" or color == "자주" or color == "검정":
                        pills.append(0)
                    elif color == "갈색|투명":
                        pills.append(1)
                    elif color == "남색" or color == "보라":
                        pills.append(2)
                    elif color == "노랑" or color == "주황":
                        pills.append(3)
                    elif color == "노랑|투명":
                        pills.append(4)
                    elif color == "보라|투명":
                        pills.append(5)
                    elif color == "분홍":
                        pills.append(6)
                    elif color == "분홍|투명":
                        pills.append(7)
                    elif color == "빨강|투명":
                        pills.append(8)
                    elif color == "연두" or color == "초록" or color == "청록":
                        pills.append(9)
                    elif color == "연두|투명":
                        pills.append(10)
                    elif color == "주황|투명":
                        pills.append(11)
                    elif color == "청록|투명":
                        pills.append(12)
                    elif color == "초록|투명":
                        pills.append(13)
                    elif color == "투명":
                        pills.append(14)
                    elif color == "파랑":
                        pills.append(15)
                    elif color == "파랑|투명":
                        pills.append(16)
                    elif color == "하양" or color == "하양|갈색":
                        pills.append(17)
                    elif color == "하양|노랑":
                        pills.append(18)
                    elif color == "하양|파랑":
                        pills.append(19)
                    elif color == "회색" or color == "하양|투명":
                        pills.append(20)

                df.insert(29, f"{project_type}_to_label", pills)
                num_classes = len(pill_type)

                print("Excel file to dataframe done!")
                return df, pill_type, num_classes

        else:
            pill_type = sorted(list(set(df[project_type])))
            print(len(pill_type))

            pills = []
            for pill in df[project_type]:
                pills.append(pill_type.index(pill))

            df.insert(29, f"{project_type}_to_label", pills)
            num_classes = len(pill_type)

            print("Excel file to dataframe done!")
            return df, pill_type, num_classes

    elif project_type == "성상":
        pill_type = ["알약", "캡슐"]

        pills = []
        for character in df["성상"]:
            # tablet: 0, capsule: 1

            ## original
            # if pill_type[1] not in character:
            #     pills.append(0)
            # else:
            #     pills.append(1)

            ## 성상에서 캡슐 & 캅셀 뽑아내기
            if "캡슐" in character or "캅셀" in character:
                if (
                    "캡슐모양" not in character
                    and "캅셀모양" not in character
                    and "캅셀형" not in character
                ):
                    pills.append(1)
                else:
                    pills.append(0)
            else:
                pills.append(0)

        ## 품목명에서 캡슐 & 캅셀 뽑아내기
        # for name in df["품목명"]:
        #     if "캡슐" in name or "캅셀" in name:
        #         pills.append(1)
        #     else:
        #         pills.append(0)

        df.insert(29, f"{project_type}_to_label", pills)
        num_classes = len(pill_type)

        print("Excel file to dataframe done!")
        return df, pill_type, num_classes

    elif project_type == "성상_의약품제형":
        is_capsule_tablet = []
        for name in df["품목명"]:
            if "캡슐" in name or "캅셀" in name:
                is_capsule_tablet.append(1)
            else:
                is_capsule_tablet.append(0)

        pill_type = [
            "알약_원형",
            "알약_장방형or타원형",
            "알약_기타",
            "알약_팔각형",
            "알약_삼각형",
            "알약_사각형",
            "알약_오각형",
            "알약_육각형",
            "알약_마름모형",
            "캡슐_장방형or타원형",
            "캡슐_기타",
        ]

        pills, i = [], 0
        for shape in df["의약품제형"]:
            ## if tablet
            if is_capsule_tablet[i] == 0:
                if shape == "원형":
                    pills.append(0)
                elif shape == "장방형" or shape == "타원형":
                    pills.append(1)
                elif shape == "기타":
                    pills.append(2)
                elif shape == "팔각형":
                    pills.append(3)
                elif shape == "삼각형":
                    pills.append(4)
                elif shape == "사각형":
                    pills.append(5)
                elif shape == "오각형":
                    pills.append(6)
                elif shape == "육각형":
                    pills.append(7)
                elif shape == "마름모형":
                    pills.append(8)
            ## if capsule
            else:
                if shape == "장방형" or shape == "타원형":
                    pills.append(9)
                elif shape == "기타":
                    pills.append(10)
            i += 1

        df.insert(29, f"{project_type}_to_label", pills)
        num_classes = len(pill_type)

        print("Excel file to dataframe done!")
        return df, pill_type, num_classes
