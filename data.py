import pandas as pd


def excel2df(excel_file_name, delete_pill_num, project_type):
    df = pd.read_excel(excel_file_name, engine="openpyxl")

    ## delete 구강정 data
    delete_nums = [200605327, 200605328, 200605329, 200605330, 200605331, 200606263]

    ## delete additional data
    delete_nums += delete_pill_num

    index_delete = []
    for delete_num in delete_nums:
        index_delete.append(df[df["품목일련번호"] == delete_num].index)

    ## delete data
    for index in index_delete:
        df = df.drop(index)

    if project_type == "의약품제형" or project_type == "색상앞":
        pill_type = sorted(list(set(df[project_type])))

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
            if pill_type[1] not in character:
                pills.append(0)
            else:
                pills.append(1)
        df.insert(29, f"{project_type}_to_label", pills)
        num_classes = len(pill_type)

        print("Excel file to dataframe done!")
        return df, pill_type, num_classes

    elif project_type == "성상_의약품제형":
        is_capsule_tablet = []
        for character in df["성상"]:
            # tablet: 0, capsule: 1
            if "캡슐" not in character:
                is_capsule_tablet.append(0)
            else:
                is_capsule_tablet.append(1)

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
