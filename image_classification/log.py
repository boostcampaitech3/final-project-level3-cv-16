import wandb


def wandb_log_train_only(train_loss, train_acc):
    wandb.log(
        {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
        }
    )


def wandb_log(
    project_type,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    best_val_loss,
    best_val_acc,
    pill_type,
    accuracy_by_label,
    custom_label=False,
):
    """
    if args.wanted_label_to_record == []:
        for i in range(len(pill_type)):
            wandb.log({f"{pill_type[i]}": accuracy_by_label[i]})
    else:
        for i in range(len(args.wanted_label_to_record)):
            wandb.log({f"{pill_type[args.wanted_label_to_record[i]]}": accuracy_by_label[args.wanted_label_to_record[i]]})

    이렇게 사용할 경우, log가 1마다 찍히는게 아니게 되어서 자동화가 힘듬
    """

    if project_type == "색상앞":
        if custom_label:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "valid_loss": val_loss,
                    "valid_accuracy": val_acc,
                    "best_loss": best_val_loss,
                    "best_accuracy": best_val_acc,
                    f"{pill_type[0]}": accuracy_by_label[0],
                    f"{pill_type[2]}": accuracy_by_label[2],
                    f"{pill_type[3]}": accuracy_by_label[3],
                    f"{pill_type[6]}": accuracy_by_label[6],
                    f"{pill_type[9]}": accuracy_by_label[9],
                    f"{pill_type[15]}": accuracy_by_label[15],
                    f"{pill_type[17]}": accuracy_by_label[17],
                    f"{pill_type[20]}": accuracy_by_label[20],
                }
            )
        else:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "valid_loss": val_loss,
                    "valid_accuracy": val_acc,
                    "best_loss": best_val_loss,
                    "best_accuracy": best_val_acc,
                    f"{pill_type[0]}": accuracy_by_label[0],
                    f"{pill_type[4]}": accuracy_by_label[4],
                    f"{pill_type[8]}": accuracy_by_label[8],
                    f"{pill_type[10]}": accuracy_by_label[10],
                    f"{pill_type[12]}": accuracy_by_label[12],
                    f"{pill_type[15]}": accuracy_by_label[15],
                    f"{pill_type[19]}": accuracy_by_label[19],
                    f"{pill_type[22]}": accuracy_by_label[22],
                    f"{pill_type[24]}": accuracy_by_label[24],
                    f"{pill_type[29]}": accuracy_by_label[29],
                }
            )

    elif project_type == "색상앞_2가지":
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc,
                "best_loss": best_val_loss,
                "best_accuracy": best_val_acc,
                f"{pill_type[0]}": accuracy_by_label[0],
                f"{pill_type[1]}": accuracy_by_label[1],
            }
        )

    elif project_type == "의약품제형":
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc,
                "best_loss": best_val_loss,
                "best_accuracy": best_val_acc,
                f"{pill_type[0]}": accuracy_by_label[0],
                f"{pill_type[1]}": accuracy_by_label[1],
                f"{pill_type[2]}": accuracy_by_label[2],
                f"{pill_type[3]}": accuracy_by_label[3],
                f"{pill_type[4]}": accuracy_by_label[4],
                f"{pill_type[5]}": accuracy_by_label[5],
                f"{pill_type[6]}": accuracy_by_label[6],
                f"{pill_type[7]}": accuracy_by_label[7],
                f"{pill_type[8]}": accuracy_by_label[8],
                f"{pill_type[9]}": accuracy_by_label[9],
            }
        )

    elif project_type == "성상":
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc,
                "best_loss": best_val_loss,
                "best_accuracy": best_val_acc,
                f"{pill_type[0]}": accuracy_by_label[0],
                f"{pill_type[1]}": accuracy_by_label[1],
            }
        )

    elif project_type == "성상_의약품제형":
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc,
                "best_loss": best_val_loss,
                "best_accuracy": best_val_acc,
                f"{pill_type[0]}": accuracy_by_label[0],
                f"{pill_type[1]}": accuracy_by_label[1],
                f"{pill_type[2]}": accuracy_by_label[2],
                f"{pill_type[3]}": accuracy_by_label[3],
                f"{pill_type[4]}": accuracy_by_label[4],
                f"{pill_type[5]}": accuracy_by_label[5],
                f"{pill_type[6]}": accuracy_by_label[6],
                f"{pill_type[7]}": accuracy_by_label[7],
                f"{pill_type[8]}": accuracy_by_label[8],
                f"{pill_type[9]}": accuracy_by_label[9],
                f"{pill_type[10]}": accuracy_by_label[10],
            }
        )
