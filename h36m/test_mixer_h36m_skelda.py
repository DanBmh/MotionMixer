import argparse
import copy
import sys
import time

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from mlp_mixer import MlpMixer

from utils.utils_mixer import delta_2_gt

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

datamode = "pred-gt"
jloss_timestep = 0

config = {
    "item_step": 1,
    # "item_step": 2,
    "window_step": 1,
    # "window_step": 40,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
# dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_4fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_4fps.json"
dataset_eval_test = dataset_eval_test.format("test")

viz_action = -1
# viz_action = 14

# ==================================================================================================


def repeat_last_timestep(input_array, num_future_timesteps):
    nbatch, _, human_joints, _ = input_array.shape
    future_timesteps = np.zeros((nbatch, num_future_timesteps, human_joints, 3))

    # Average of deltas of last input
    last_input = np.mean(np.mean(input_array[:, -1:], axis=1), axis=1)
    last_input = np.repeat(np.expand_dims(last_input, axis=1), 15, axis=1)
    # # Just last input
    # last_input = input_array[:, -1, :, :]
    # # Torso average delta
    # last_input = np.mean(np.mean(input_array[:, -1:, [0,1,4,8,11,14], :], axis=1), axis=1)
    # last_input = np.repeat(np.expand_dims(last_input, axis=1), 15, axis=1)

    for i in range(nbatch):
        for j in range(human_joints):
            for coord in range(3):
                future_timesteps[i, :, j, coord] = (
                    np.ones(num_future_timesteps) * last_input[i, j, coord]
                )

    return future_timesteps


# ==================================================================================================


def calc_delta(sequences_train, sequences_gt, args):
    # print(sequences_train.shape, sequences_gt.shape)
    sequences_all = torch.cat((sequences_train, sequences_gt), 1)
    sequences_all_delta = [sequences_all[:, 1, :] - sequences_all[:, 0, :]]
    for i in range(args.input_n + args.output_n - 1):
        sequences_all_delta.append(sequences_all[:, i + 1, :] - sequences_all[:, i, :])

    sequences_all_delta = torch.stack((sequences_all_delta)).permute(1, 0, 2)
    sequences_train_delta = sequences_all_delta[:, 0 : args.input_n, :]
    return sequences_train_delta


# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device, dmode):
    sequences = utils_pipeline.make_input_sequence(batch, split, dmode)

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def viz_joints_3d(sequences_predict, batch):
    batch = batch[0]

    if datamode.split("-")[0] == "gt":
        last_input_pose = batch["input"][-1]["bodies3D"][0]
    else:
        last_input_pose = batch["input"][-1]["predictions"][0]

    vis_seq_pred = (
        sequences_predict.cpu()
        .detach()
        .numpy()
        .reshape(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
    )[0]
    utils_pipeline.visualize_pose_trajectories(
        np.array([cs["bodies3D"][0] for cs in batch["input"]]),
        # np.array([]),
        np.array([cs["bodies3D"][0] for cs in batch["target"]])[::1],
        utils_pipeline.make_absolute_with_last_input(vis_seq_pred, last_input_pose)[
            ::1
        ],
        batch["joints"],
        # {"room_size": [3200, 4800, 2000], "room_center": [0, 0, 1000]},
        {},
    )

    plt.grid(False)
    plt.axis("off")
    plt.show()


# ==================================================================================================


def run_test(model, args):

    device = args.dev
    model.eval()

    config["input_n"] = args.input_n
    config["output_n"] = args.output_n

    # Load preprocessed datasets
    cfg = copy.deepcopy(config)
    if "mocap" in dataset_eval_test:
        cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
    dataset_test, dlen = utils_pipeline.load_dataset(dataset_eval_test, "test", cfg)
    dataset_test = dataset_test["sequences"]
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, config)

    stime = time.time()
    frame_losses = np.zeros([args.output_n])
    # frame_losses = np.zeros([args.input_n])
    joint_losses = np.zeros([len(config["select_joints"])])
    max_jlosses = np.zeros([len(config["select_joints"])])
    nitems = 0

    with torch.no_grad():
        nbatch = 1

        for batch in tqdm.tqdm(label_gen_test, total=dlen):

            if nbatch == 1:
                batch = [batch]

            if viz_action != -1 and viz_action != batch[0]["action"]:
                continue

            nitems += nbatch
            sequences_train = prepare_sequences(
                batch, nbatch, "input", device, datamode
            )
            sequences_gt = prepare_sequences(batch, nbatch, "target", device, datamode)

            if args.delta_x:
                sequences_train_delta = calc_delta(sequences_train, sequences_gt, args)
                sequences_predict = model(sequences_train_delta)
                sequences_predict = delta_2_gt(
                    sequences_predict, sequences_train[:, -1, :]
                )

            else:
                sequences_predict = model(sequences_train)

            # # Uncomment this to calculate input pose errors
            # sequences_predict = prepare_sequences(
            #     batch, nbatch, "input", device, "pred-pred"
            # )
            # sequences_gt = prepare_sequences(batch, nbatch, "input", device, "gt-gt")
            # sequences_predict = sequences_predict.reshape(
            #     nbatch, -1, args.pose_dim // 3, 3
            # )
            # sequences_gt = sequences_gt.reshape(nbatch, -1, args.pose_dim // 3, 3)
            # last_input_pose_gt = batch[0]["input"][-1]["bodies3D"][0]
            # last_input_pose_pred = batch[0]["input"][-1]["predictions"][0]
            # sequences_predict = sequences_predict.cpu().data.numpy()
            # sequences_gt = sequences_gt.cpu().data.numpy()
            # sequences_predict = utils_pipeline.make_absolute_with_last_input(
            #     sequences_predict, last_input_pose_pred
            # )
            # sequences_gt = utils_pipeline.make_absolute_with_last_input(
            #     sequences_gt, last_input_pose_gt
            # )
            # sequences_predict = torch.from_numpy(sequences_predict).float().to(device)
            # sequences_gt = torch.from_numpy(sequences_gt).float().to(device)

            # # Uncomment this to run a test which only predicts the last known timestep
            # sequences_train_delta = calc_delta(sequences_train, sequences_gt, args)
            # seq_train_np = sequences_train_delta.cpu().data.numpy()
            # # seq_train_np = sequences_train.cpu().data.numpy()
            # seq_train_np = seq_train_np.reshape(nbatch, -1, args.pose_dim // 3, 3)
            # seq_pred_np = repeat_last_timestep(seq_train_np, args.output_n)
            # seq_pred_np = seq_pred_np.reshape(nbatch, args.output_n, -1)
            # sequences_predict = torch.from_numpy(seq_pred_np).float().to(device)
            # sequences_predict = delta_2_gt(sequences_predict, sequences_train[:, -1, :])

            if viz_action != -1:
                viz_joints_3d(sequences_predict, batch)

            sequences_predict = sequences_predict.reshape(
                nbatch, -1, args.pose_dim // 3, 3
            )
            sequences_gt = sequences_gt.reshape(nbatch, -1, args.pose_dim // 3, 3)

            loss = torch.sqrt(
                torch.sum((sequences_predict - sequences_gt) ** 2, dim=-1)
            )
            floss = torch.sum(torch.mean(loss, dim=2), dim=0)
            jloss = torch.sum(loss[:, jloss_timestep, :], dim=0)
            frame_losses += floss.cpu().data.numpy()
            joint_losses += jloss.cpu().data.numpy()
            max_jlosses = np.maximum(max_jlosses, jloss.cpu().data.numpy())

    avg_losses = frame_losses / nitems
    avg_jlosses = joint_losses / nitems
    avg_jlosses = np.round(avg_jlosses).astype(int)
    max_jlosses = np.round(max_jlosses).astype(int)
    avg_losses = np.round(avg_losses, 1)
    print("Averaged frame losses in mm are:", avg_losses)
    print(
        "Averaged joint losses at timestep", jloss_timestep, "in mm are:", avg_jlosses
    )
    print("Maximum joint losses at timestep", jloss_timestep, "in mm are:", max_jlosses)

    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))


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
    model.load_state_dict(torch.load(args.model_path))

    run_test(model, args)
