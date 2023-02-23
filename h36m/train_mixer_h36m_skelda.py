import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import tqdm
from mlp_mixer import MlpMixer
from torch.utils.tensorboard import SummaryWriter

from utils.utils_mixer import delta_2_gt, mpjpe_error

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

datapath_save_out = "/datasets/tmp/human36m/{}_forecast_samples.json"
config = {
    "item_step": 2,
    "window_step": 2,
    "input_n": 50,
    "output_n": 25,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        # "middlefoot_right",
        # "forefoot_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        # "middlefoot_left",
        # "forefoot_left",
        # "spine_upper",
        # "neck",
        "nose",
        # "head",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        # "hand_left",
        # "thumb_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        # "hand_right",
        # "thumb_right",
        "shoulder_middle",
    ],
}

# ==================================================================================================


def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs) < 2:
        log_dir = os.path.join(out_dir, "exp0")
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, "exp%i" % (len(dirs) - 1))
        os.mkdir(log_dir)

    return log_dir


# ==================================================================================================


def calc_delta(sequences_train, sequences_gt, args):
    sequences_all = torch.cat((sequences_train, sequences_gt), 1)
    sequences_all_delta = [sequences_all[:, 1, :] - sequences_all[:, 0, :]]
    for i in range(args.input_n + args.output_n - 1):
        sequences_all_delta.append(sequences_all[:, i + 1, :] - sequences_all[:, i, :])

    sequences_all_delta = torch.stack((sequences_all_delta)).permute(1, 0, 2)
    sequences_train_delta = sequences_all_delta[:, 0 : args.input_n, :]
    return sequences_train_delta


# ==================================================================================================


def run_train(model, model_name, args):

    log_dir = get_log_dir(args.root)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print("Save data of the run in: %s" % log_dir)

    device = args.dev
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = utils_pipeline.load_dataset(
        datapath_save_out, "train", config
    )
    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        datapath_save_out, "eval", config
    )

    train_loss, val_loss, test_loss = [], [], []
    best_loss = np.inf

    for epoch in range(args.n_epochs):
        print("Run epoch: %i" % epoch)
        running_loss = 0
        model.train()

        label_gen_train = utils_pipeline.create_labels_generator(dataset_train, config)
        label_gen_eval = utils_pipeline.create_labels_generator(dataset_eval, config)

        nbatch = args.batch_size
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen_train, batch_size=nbatch),
            total=int(dlen_train / nbatch),
        ):

            sequences_train = utils_pipeline.make_input_sequence(batch, "input")
            sequences_gt = utils_pipeline.make_input_sequence(batch, "target")

            augment = True
            if augment:
                sequences_train, sequences_gt = utils_pipeline.apply_augmentations(
                    sequences_train, sequences_gt
                )

            # Merge joints and coordinates to a single dimension
            sequences_train = sequences_train.reshape(
                [nbatch, sequences_train.shape[1], -1]
            )
            sequences_gt = sequences_gt.reshape([nbatch, sequences_gt.shape[1], -1])

            sequences_train = torch.from_numpy(sequences_train).to(device)
            sequences_gt = torch.from_numpy(sequences_gt).to(device)

            optimizer.zero_grad()

            if args.delta_x:
                sequences_train_delta = calc_delta(sequences_train, sequences_gt, args)
                sequences_predict = model(sequences_train_delta)
                sequences_predict = delta_2_gt(
                    sequences_predict, sequences_train[:, -1, :]
                )

            else:
                sequences_predict = model(sequences_train)

            loss = mpjpe_error(sequences_predict, sequences_gt)
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            running_loss += loss * nbatch
        train_loss.append(
            running_loss.detach().cpu() / (int(dlen_train / nbatch) * nbatch)
        )

        if args.use_scheduler:
            scheduler.step()

        eval_loss = run_eval(model, label_gen_eval, dlen_eval, args)
        val_loss.append(eval_loss)

        tb_writer.add_scalar("loss/train", train_loss[-1].item(), epoch)
        tb_writer.add_scalar("loss/val", val_loss[-1].item(), epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

        if eval_loss < best_loss:
            best_loss = eval_loss
            print("New best validation loss: %f" % best_loss)
            print("Saving best model...")
            torch.save(model.state_dict(), os.path.join(args.model_path, model_name))


# ==================================================================================================


def run_eval(model, dataset_gen_eval, dlen_eval, args):

    device = args.dev
    model.eval()

    with torch.no_grad():
        running_loss = 0

        nbatch = args.batch_size_test
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(dataset_gen_eval, batch_size=nbatch),
            total=int(dlen_eval / nbatch),
        ):

            sequences_train = utils_pipeline.make_input_sequence(batch, "input")
            sequences_gt = utils_pipeline.make_input_sequence(batch, "target")

            # Merge joints and coordinates to a single dimension
            sequences_train = sequences_train.reshape(
                [nbatch, sequences_train.shape[1], -1]
            )
            sequences_gt = sequences_gt.reshape([nbatch, sequences_gt.shape[1], -1])

            sequences_train = torch.from_numpy(sequences_train).to(device)
            sequences_gt = torch.from_numpy(sequences_gt).to(device)

            if args.delta_x:
                sequences_train_delta = calc_delta(sequences_train, sequences_gt, args)
                sequences_predict = model(sequences_train_delta)
                sequences_predict = delta_2_gt(
                    sequences_predict, sequences_train[:, -1, :]
                )

            else:
                sequences_predict = model(sequences_train)

            loss = mpjpe_error(sequences_predict, sequences_gt)
            running_loss += loss * nbatch

        avg_loss = running_loss.detach().cpu() / (int(dlen_eval / nbatch) * nbatch)
        print("overall average loss in mm is: {:.3f}".format(avg_loss))
        return avg_loss


# ==================================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data_h36m/",
        help="path to the unziped dataset directories(H36m/AMASS/3DPW)",
    )
    parser.add_argument(
        "--input_n", type=int, default=10, help="number of model's input frames"
    )
    parser.add_argument(
        "--output_n", type=int, default=25, help="number of model's output frames"
    )
    parser.add_argument(
        "--skip_rate",
        type=int,
        default=5,
        choices=[1, 5],
        help="rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW",
    )
    parser.add_argument(
        "--num_worker", default=4, type=int, help="number of workers in the dataloader"
    )
    parser.add_argument(
        "--root", default="./runs", type=str, help="root path for the logging"
    )  #'./runs'

    parser.add_argument("--activation", default="mish", type=str, required=False)
    parser.add_argument("--r_se", default=8, type=int, required=False)

    parser.add_argument("--n_epochs", default=50, type=int, required=False)
    parser.add_argument("--batch_size", default=50, type=int, required=False)
    parser.add_argument("--loader_shuffle", default=True, type=bool, required=False)
    parser.add_argument("--pin_memory", default=False, type=bool, required=False)
    parser.add_argument("--loader_workers", default=4, type=int, required=False)
    parser.add_argument("--load_checkpoint", default=False, type=bool, required=False)
    parser.add_argument("--dev", default="cuda:0", type=str, required=False)
    parser.add_argument(
        "--initialization",
        type=str,
        default="none",
        help="none, glorot_normal, glorot_uniform, hee_normal, hee_uniform",
    )
    parser.add_argument("--use_scheduler", default=True, type=bool, required=False)
    parser.add_argument(
        "--milestones",
        type=list,
        default=[15, 25, 35, 40],
        help="the epochs after which the learning rate is adjusted by gamma",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="gamma correction to the learning rate, after reaching the milestone epochs",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        help="select max norm to clip gradients",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/h36m",
        help="directory with the models checkpoints ",
    )
    parser.add_argument(
        "--actions_to_consider",
        default="all",
        help="Actions to visualize.Choose either all or a list of actions",
    )
    parser.add_argument(
        "--batch_size_test", type=int, default=256, help="batch size for the test set"
    )
    parser.add_argument(
        "--visualize_from",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="choose data split to visualize from(train-val-test)",
    )
    parser.add_argument(
        "--loss_type", type=str, default="mpjpe", choices=["mpjpe", "angle"]
    )
    parser.add_argument("--hidden_dim", default=50, type=int, required=False)
    parser.add_argument("--num_blocks", default=4, type=int, required=False)
    parser.add_argument("--tokens_mlp_dim", default=20, type=int, required=False)
    parser.add_argument("--channels_mlp_dim", default=50, type=int, required=False)
    parser.add_argument("--regularization", default=0.1, type=float, required=False)
    parser.add_argument("--pose_dim", default=45, type=int, required=False)
    parser.add_argument(
        "--delta_x",
        type=bool,
        default=True,
        help="predicting the difference between 2 frames",
    )
    parser.add_argument("--lr", default=0.001, type=float, required=False)

    args = parser.parse_args()
    print(args)

    model = MlpMixer(
        num_classes=args.pose_dim,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        tokens_mlp_dim=args.tokens_mlp_dim,
        channels_mlp_dim=args.channels_mlp_dim,
        seq_len=args.input_n,
        pred_len=args.output_n,
        activation=args.activation,
        mlp_block_type="normal",
        regularization=args.regularization,
        input_size=args.pose_dim,
        initialization="none",
        r_se=args.r_se,
        use_max_pooling=False,
        use_se=True,
    )

    model = model.to(args.dev)
    print(
        "total number of parameters of the network is: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    model_name = "h36_3d_abs_" + str(args.output_n) + "frames_ckpt3"

    stime = time.time()
    run_train(model, model_name, args)

    ftime = time.time()
    print("Training took {} seconds".format(int(ftime - stime)))
