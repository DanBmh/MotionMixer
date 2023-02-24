import argparse
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

viz_action = ""
# viz_action = "walking"

# ==================================================================================================


def repeat_last_timestep(input_array, num_future_timesteps):
    nbatch, _, human_joints, _ = input_array.shape
    future_timesteps = np.zeros((nbatch, num_future_timesteps, human_joints, 3))

    for i in range(nbatch):
        for j in range(human_joints):
            for coord in range(3):
                future_timesteps[i, :, j, coord] = (
                    np.ones(num_future_timesteps) * input_array[i, -1, j, coord]
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


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split)

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def viz_joints_3d(sequences_predict, batch):
    batch = batch[0]
    vis_seq_pred = (
        sequences_predict.cpu()
        .detach()
        .numpy()
        .reshape(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
    )[0]
    utils_pipeline.visualize_pose_trajectories(
        np.array([cs["bodies3D"][0] for cs in batch["input"]]),
        np.array([cs["bodies3D"][0] for cs in batch["target"]]),
        utils_pipeline.make_absolute_with_last_input(vis_seq_pred, batch),
        batch["joints"],
        {"room_size": [3200, 4800, 2000], "room_center": [0, 0, 1000]},
    )
    plt.show()


# ==================================================================================================


def run_test(model, args):

    device = args.dev
    model.eval()

    # Load preprocessed datasets
    dataset_test, dlen = utils_pipeline.load_dataset(datapath_save_out, "test", config)
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, config)

    stime = time.time()
    frame_losses = np.zeros([args.output_n])
    nitems = 0

    with torch.no_grad():
        nbatch = 1

        for batch in tqdm.tqdm(label_gen_test, total=dlen):

            if nbatch == 1:
                batch = [batch]

            if viz_action != "" and viz_action != batch[0]["action"]:
                continue

            nitems += nbatch
            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)

            if args.delta_x:
                sequences_train_delta = calc_delta(sequences_train, sequences_gt, args)
                sequences_predict = model(sequences_train_delta)
                sequences_predict = delta_2_gt(
                    sequences_predict, sequences_train[:, -1, :]
                )

            else:
                sequences_predict = model(sequences_train)

            # # Uncomment this to run a test which only predicts the last known timestep
            # seq_train_np = sequences_train.cpu().data.numpy()
            # seq_train_np = seq_train_np.reshape(nbatch, -1, args.pose_dim // 3, 3)
            # seq_pred_np = repeat_last_timestep(seq_train_np, args.output_n)
            # seq_pred_np = seq_pred_np.reshape(nbatch, args.output_n, -1)
            # sequences_predict = torch.from_numpy(seq_pred_np).float().to(device)

            if viz_action != "":
                viz_joints_3d(sequences_predict, batch)

            loss = torch.sqrt(
                torch.sum(
                    (
                        sequences_predict.view(nbatch, -1, args.pose_dim // 3, 3)
                        - sequences_gt.view(nbatch, -1, args.pose_dim // 3, 3)
                    )
                    ** 2,
                    dim=-1,
                )
            )
            loss = torch.sum(torch.mean(loss, dim=2), dim=0)
            frame_losses += loss.cpu().data.numpy()

    avg_losses = frame_losses / nitems
    print("Averaged frame losses in mm are:", avg_losses)

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
    # parser.add_argument("--hidden_dim", default=512, type=int, required=False)
    # parser.add_argument("--num_blocks", default=8, type=int, required=False)
    # parser.add_argument("--tokens_mlp_dim", default=512, type=int, required=False)
    # parser.add_argument("--channels_mlp_dim", default=512, type=int, required=False)
    # parser.add_argument("--hidden_dim", default=100, type=int, required=False)
    # parser.add_argument("--num_blocks", default=6, type=int, required=False)
    # parser.add_argument("--tokens_mlp_dim", default=100, type=int, required=False)
    # parser.add_argument("--channels_mlp_dim", default=100, type=int, required=False)
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
    model.load_state_dict(torch.load(args.model_path))

    run_test(model, args)
