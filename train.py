import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
import timm
import argparse
import wandb
import os
import random

from tqdm import tqdm

from data import excel2df
from dataset import PillDataset
from log import wandb_log
from test import inference


def customize_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    if args.project_name:
        wandb.init(
            project="final-project",
            entity="medic",
            name=f"{args.user_name}_{args.model_name}_{args.project_name}",
        )
    else:
        wandb.init(
            project="final-project",
            entity="medic",
            name=f"{args.user_name}_{args.model_name}_{args.project_type}",
        )

    customize_seed(args.seed)

    df, pill_type, num_classes = excel2df(
        args.excel_file_name, args.delete_pill_num, args.project_type, args.custom_label
    )

    if args.create_test_data:
        val_df, test_df, train_loader, val_loader, test_loader = PillDataset(
            df,
            args.project_type,
            args.batch_size,
            args.image_file_path,
            args.create_test_data,
        )
    else:
        val_df, train_loader, val_loader = PillDataset(
            df,
            args.project_type,
            args.batch_size,
            args.image_file_path,
            args.create_test_data,
        )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if args.opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.sch == "StepLR":
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    elif args.sch == "Cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)

    if args.project_name:
        name = f"{args.model_name}_{args.project_name}"
    else:
        name = f"{args.model_name}_{args.project_type}"

    os.makedirs(os.path.join(os.getcwd(), "results", name), exist_ok=True)

    counter = 0
    best_val_acc, best_val_loss = 0, np.inf

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value, matches = 0, 0

        for idx, train_batch in tqdm(enumerate(train_loader)):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()

            # -- Gradient Accumulation
            if (idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.train_log_interval == 0:
                train_loss = loss_value / args.train_log_interval
                train_acc = matches / args.batch_size / args.train_log_interval
                current_lr = scheduler.get_last_lr()
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items, val_acc_items = [], []
            label_accuracy, total_label = [0] * num_classes, [0] * num_classes

            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                ## label별 accuracy
                for i in range(len(labels)):
                    total_label[int(labels[i])] += 1
                    if labels[i] == preds[i]:
                        label_accuracy[int(labels[i])] += 1

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_df)

            # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model..")
                torch.save(
                    model.state_dict(),
                    f"results/{name}/best.ckpt",
                )
                best_val_acc = val_acc
                counter = 0
            else:
                counter += 1
            # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > args.patience:
                print("Early Stopping...")
                break

            ## 파이썬 배열 나눗셈 https://bearwoong.tistory.com/60
            accuracy_by_label = np.array(label_accuracy) / np.array(total_label)

            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )

        ## 기존의 학습과 다른 label을 사용한다면, 무조건 wandb.py 안의 wandb.log()를 수정해야 함
        wandb_log(
            args.project_type,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            best_val_loss,
            best_val_acc,
            pill_type,
            accuracy_by_label,
            args.custom_label,
        )

    if args.create_test_data:
        return test_df, test_loader, model, device, pill_type
    else:
        return val_df, val_loader, model, device, pill_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping (default: 10)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate (defalt: 0.0001)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=5,
        help="learning rate deacy step (default: 5)",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
        help="training accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--train_log_interval",
        type=int,
        default=100,
        help="training log interval (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=16, help="fix seed (default: 16)")
    parser.add_argument(
        "--opt", type=str, default="Adam", help="optimizer (default: Adam)"
    )
    parser.add_argument(
        "--sch", type=str, default="StepLR", help="scheduler (default: StepLR)"
    )

    ## path, type, and name
    parser.add_argument(
        "--excel_file_name",
        type=str,
        default="./pill_excel_data/OpenData_PotOpenTabletIdntfc20220412.xls",
        help="name of the pill data excel (default: ./pill_excel_data/OpenData_PotOpenTabletIdntfc20220412.xls)",
    )
    parser.add_argument(
        "--image_file_path",
        type=str,
        default="/opt/ml/final-project-level3-cv-16/data/raw_data",
        help="path to image file (default: /opt/ml/final-project-level3-cv-16/data/raw_data)",
    )
    parser.add_argument(
        "--project_type",
        type=str,
        default="shape",
        help="which column to use (default: shape)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="YH",
        help="user name (default: YH)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="timm model name (default: resnet18)",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="customize project name of what difference the project has (default: None)",
    )
    parser.add_argument(
        "--create_test_data",
        type=bool,
        default=False,
        help="create test data from validation data (default: False)",
    )

    ## customize data
    parser.add_argument(
        "--delete_pill_num",
        type=int,
        nargs="+",
        default=[],
        help="list of file to delete (default: [])",
    )
    parser.add_argument(
        "--custom_label",
        type=bool,
        default=False,
        help="customize labels for training (default: False)",
    )

    args = parser.parse_args()

    train(args)

    # if args.create_test_data:
    #     test_df, test_loader, model, device, pill_type = train(args)
    #     inference(test_df, test_loader, model, device, pill_type)
    # else:
    #     val_df, val_loader, model, device, pill_type = train(args)
    #     inference(val_df, val_loader, model, device, pill_type)
