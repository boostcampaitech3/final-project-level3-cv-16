import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, df, project_type, image_file_path, transform=None):
        super().__init__()
        self.df = df.reset_index()
        self.image_id = self.df["품목일련번호"]
        self.labels = self.df[f"{project_type}_to_label"]
        self.image_file_path = image_file_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_id[idx]
        label = self.labels[idx]
        image_path = f"{self.image_file_path}/{image_id}.jpg"
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label


def PillDataset(df, project_type, batch_size, image_file_path, create_test_data=False):
    ## Split Train/Val
    image_num = df["품목일련번호"]
    label = df[f"{project_type}_to_label"]

    if create_test_data:
        x_train, x_tmp, y_train, y_tmp = train_test_split(
            image_num, label, test_size=0.2, stratify=label, random_state=22
        )

        x_valid, x_test, y_valid, y_test = train_test_split(
            x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=22
        )

        train_zip = zip(x_train, y_train)
        train_df = pd.DataFrame(train_zip)
        train_df.columns = ["품목일련번호", f"{project_type}_to_label"]

        val_zip = zip(x_valid, y_valid)
        val_df = pd.DataFrame(val_zip)
        val_df.columns = ["품목일련번호", f"{project_type}_to_label"]

        test_zip = zip(x_test, y_test)
        test_df = pd.DataFrame(test_zip)
        test_df.columns = ["품목일련번호", f"{project_type}_to_label"]

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        train_dataset = CustomDataset(
            train_df, project_type, image_file_path, transform=train_transform
        )
        val_dataset = CustomDataset(
            val_df, project_type, image_file_path, transform=valid_transform
        )
        test_dataset = CustomDataset(
            test_df, project_type, image_file_path, transform=test_transform
        )

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        print("Data loading done!")
        return val_df, test_df, train_loader, val_loader, test_loader

    else:
        x_train, x_valid, y_train, y_valid = train_test_split(
            image_num, label, test_size=0.2, stratify=label, random_state=22
        )

        train_zip = zip(x_train, y_train)
        train_df = pd.DataFrame(train_zip)
        train_df.columns = ["품목일련번호", f"{project_type}_to_label"]

        val_zip = zip(x_valid, y_valid)
        val_df = pd.DataFrame(val_zip)
        val_df.columns = ["품목일련번호", f"{project_type}_to_label"]

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        train_dataset = CustomDataset(
            train_df, project_type, image_file_path, transform=train_transform
        )
        val_dataset = CustomDataset(
            val_df, project_type, image_file_path, transform=valid_transform
        )

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

        print("Data loading done!")
        return val_df, train_loader, val_loader
