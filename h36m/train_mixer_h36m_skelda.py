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

datamode = "gt-gt"

# datapath_save_out = "/datasets/tmp/human36m/{}_forecast_samples.json"
datapath_save_out = "/datasets/tmp/human36m/{}_forecast_kppspose.json"
config = {
    # "item_step": 2,
    "item_step": 5,
    "window_step": 2,
    # "input_n": 50,
    # "output_n": 25,
    # "input_n": 20,
    # "output_n": 10,
    "input_n": 60,
    "output_n": 30,
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

# datapath_save_out = "/datasets/tmp/mocap/{}_forecast_samples.json"
# config = {
#     # "item_step": 2,
#     "item_step": 3,
#     "window_step": 2,
#     # "input_n": 30,
#     # "output_n": 15,
#     # "input_n": 20,
#     # "output_n": 10,
#     "input_n": 60,
#     "output_n": 30,
#     "select_joints": [
#         "hip_middle",
#         # "spine_lower",
#         "hip_right",
#         "knee_right",
#         "ankle_right",
#         # "middlefoot_right",
#         # "forefoot_right",
#         "hip_left",
#         "knee_left",
#         "ankle_left",
#         # "middlefoot_left",
#         # "forefoot_left",
#         # "spine2",
#         # "spine3",
#         # "spine_upper",
#         # "neck",
#         # "head_lower",
#         "head_upper",
#         "shoulder_right",
#         "elbow_right",
#         "wrist_right",
#         # "hand_right1",
#         # "hand_right2",
#         # "hand_right3",
#         # "hand_right4",
#         "shoulder_left",
#         "elbow_left",
#         "wrist_left",
#         # "hand_left1",
#         # "hand_left2",
#         # "hand_left3",
#         # "hand_left4"
#         "shoulder_middle",
#     ],
# }

limbs = [
    ("shoulder_left", "shoulder_right"),
    ("shoulder_left", "elbow_left"),
    ("shoulder_right", "elbow_right"),
    ("elbow_left", "wrist_left"),
    ("elbow_right", "wrist_right"),
    ("hip_middle", "shoulder_middle"),
    ("hip_left", "hip_right"),
    ("hip_left", "knee_left"),
    ("hip_right", "knee_right"),
    ("knee_left", "ankle_left"),
    ("knee_right", "ankle_right"),
]
limb_ids = [
    (config["select_joints"].index(s1), config["select_joints"].index(s2))
    for (s1, s2) in limbs
]
limb_starts = [s1 for (s1, _) in limb_ids]
limb_ends = [s2 for (_, s2) in limb_ids]

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


def scale_error(sequences_predict, sequences_gt):
    stshape = sequences_predict.shape
    sgshape = sequences_gt.shape
    sequences_predict = sequences_predict.reshape(stshape[0], stshape[1], -1, 3)
    sequences_gt = sequences_gt.reshape(sgshape[0], sgshape[1], -1, 3)

    # Calculate lengths of the limbs
    lengths_gt = sequences_gt[:, :, limb_starts, :] - sequences_gt[:, :, limb_ends, :]
    lengths_pd = (
        sequences_predict[:, :, limb_starts, :] - sequences_predict[:, :, limb_ends, :]
    )
    lengths_gt = torch.sqrt(torch.sum(lengths_gt**2, -1))
    lengths_pd = torch.sqrt(torch.sum(lengths_pd**2, -1))

    error = 0.1 * torch.mean((lengths_pd - lengths_gt) ** 2)
    return error


# ==================================================================================================


def run_train(model, model_path, args):

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
    esplit = "test" if "mocap" in datapath_save_out else "eval"
    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        datapath_save_out, esplit, config
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

            sequences_train = utils_pipeline.make_input_sequence(
                batch, "input", datamode
            )
            sequences_gt = utils_pipeline.make_input_sequence(batch, "target", datamode)

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
            # loss += scale_error(sequences_predict, sequences_gt)
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
            torch.save(model.state_dict(), model_path)


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

            sequences_train = utils_pipeline.make_input_sequence(
                batch, "input", datamode
            )
            sequences_gt = utils_pipeline.make_input_sequence(batch, "target", datamode)

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
            # loss += scale_error(sequences_predict, sequences_gt)
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
        default="./checkpoints/h36m/h36_3d_abs_25frames_ckpt",
        help="directory with the models checkpoints ",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="",
        help="directory with the model weights to copy",
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
    parser.add_argument("--hidden_dim", default=512, type=int, required=False)
    parser.add_argument("--num_blocks", default=8, type=int, required=False)
    parser.add_argument("--tokens_mlp_dim", default=512, type=int, required=False)
    parser.add_argument("--channels_mlp_dim", default=512, type=int, required=False)
    # parser.add_argument("--hidden_dim", default=100, type=int, required=False)
    # parser.add_argument("--num_blocks", default=6, type=int, required=False)
    # parser.add_argument("--tokens_mlp_dim", default=100, type=int, required=False)
    # parser.add_argument("--channels_mlp_dim", default=100, type=int, required=False)
    # parser.add_argument("--hidden_dim", default=50, type=int, required=False)
    # parser.add_argument("--num_blocks", default=4, type=int, required=False)
    # parser.add_argument("--tokens_mlp_dim", default=20, type=int, required=False)
    # parser.add_argument("--channels_mlp_dim", default=50, type=int, required=False)
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

    if args.model_weights_path != "":
        print("Loading model weights from:", args.model_weights_path)
        model.load_state_dict(torch.load(args.model_weights_path))

    stime = time.time()
    run_train(model, args.model_path, args)

    ftime = time.time()
    print("Training took {} seconds".format(int(ftime - stime)))
