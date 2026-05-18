import argparse
import logging
import os
import pickle
import pprint
import sys
import time
from collections import defaultdict

import numpy as np
import torch

import utils
from average_meter import AverageMeter
from model.hidden_moe import HiddenMoE
from options_moe import HiDDenMoEConfiguration, TrainingOptions


def get_data_loaders(hidden_config, train_options, num_workers=0):
    train_loader, val_loader = utils.get_data_loaders(hidden_config, train_options)
    train_loader.num_workers = num_workers
    val_loader.num_workers = num_workers
    return train_loader, val_loader


def log_progress_and_flush(losses_accu):
    utils.log_progress(losses_accu)
    sys.stdout.flush()


def train(
    model: HiddenMoE,
    device: torch.device,
    hidden_config: HiDDenMoEConfiguration,
    train_options: TrainingOptions,
    this_run_folder: str,
    tb_logger,
    print_each: int = 3,
    num_workers: int = 0,
):
    train_data, val_data = get_data_loaders(hidden_config, train_options, num_workers=num_workers)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    images_to_save = 8
    saved_images_size = (512, 512)
    lr_reduced = False
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        model.set_epoch(epoch)
        if epoch > 80 and not lr_reduced:
            for optim in (model.optimizer_enc_dec, model.optimizer_discrim):
                for param_group in optim.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.1
            lr_reduced = True
            logging.info("Phase 3 LR reduction applied (x0.1).")
            sys.stdout.flush()

        logging.info("\nStarting epoch {}/{}".format(epoch, train_options.number_of_epochs))
        logging.info(
            "Batch size = {} | Steps in epoch = {} | Log every {} steps | GPUs = {}".format(
                train_options.batch_size, steps_in_epoch, print_each, max(gpu_count, 1)
            )
        )
        sys.stdout.flush()
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1

        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, _ = model.train_on_batch([image, message])

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step == 1 or step % print_each == 0 or step == steps_in_epoch:
                logging.info("Epoch: {}/{} Step: {}/{}".format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                log_progress_and_flush(training_losses)
                logging.info("-" * 40)
                sys.stdout.flush()
            step += 1

        if epoch >= 20 and training_losses["expert_max_use "].avg > 0.4:
            logging.warning("Expert usage warning: max expert share is {:.4f} (> 0.40).".format(
                training_losses["expert_max_use "].avg
            ))

        train_duration = time.time() - epoch_start
        logging.info("Epoch {} training duration {:.2f} sec".format(epoch, train_duration))
        logging.info("-" * 40)
        utils.write_losses(os.path.join(this_run_folder, "train.csv"), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info("Running validation for epoch {}/{}".format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages, router_probs) = model.validate_on_batch([image, message])
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(
                    image.cpu()[:images_to_save, :, :, :],
                    encoded_images[:images_to_save, :, :, :].cpu(),
                    epoch,
                    os.path.join(this_run_folder, "images"),
                    resize_to=saved_images_size,
                )
                first_iteration = False

        log_progress_and_flush(validation_losses)
        logging.info("-" * 40)
        sys.stdout.flush()
        model.save_checkpoint(
            train_options.experiment_name,
            epoch,
            os.path.join(this_run_folder, "checkpoints"),
        )
        utils.write_losses(os.path.join(this_run_folder, "validation.csv"), validation_losses, epoch, time.time() - epoch_start)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description="Training of HiDDeN MoE nets")
    subparsers = parser.add_subparsers(dest="command", help="Sub-parser for commands")

    new_run_parser = subparsers.add_parser("new", help="Starts a new MoE run")
    new_run_parser.add_argument("--data-dir", "-d", required=True, type=str, help="The directory where the data is stored.")
    new_run_parser.add_argument("--batch-size", "-b", required=True, type=int, help="The batch size.")
    new_run_parser.add_argument("--epochs", "-e", default=200, type=int, help="Number of epochs.")
    new_run_parser.add_argument("--name", required=True, type=str, help="Experiment name.")
    new_run_parser.add_argument("--size", "-s", default=128, type=int, help="Image size.")
    new_run_parser.add_argument("--message", "-m", default=30, type=int, help="Message length in bits.")
    new_run_parser.add_argument("--num-experts", default=8, type=int, help="Number of experts.")
    new_run_parser.add_argument("--top-k", default=2, type=int, help="Top-k routing.")
    new_run_parser.add_argument("--balance-loss-weight", default=0.01, type=float, help="MoE balance loss weight.")
    new_run_parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard logging.")
    new_run_parser.add_argument("--enable-fp16", dest="enable_fp16", action="store_true", help="Enable mixed precision.")
    new_run_parser.add_argument(
        "--print-each",
        default=3,
        type=int,
        help="Log training progress every N steps (default: 3).",
    )
    new_run_parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="DataLoader workers (use 0 on Kaggle notebooks).",
    )
    new_run_parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable DataParallel even if multiple GPUs are available.",
    )
    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    continue_parser = subparsers.add_parser("continue", help="Continue an existing MoE run")
    continue_parser.add_argument("--folder", "-f", required=True, type=str, help="Run folder path.")
    continue_parser.add_argument("--data-dir", "-d", required=False, type=str, help="Optional data dir override.")
    continue_parser.add_argument("--epochs", "-e", required=False, type=int, help="Optional epoch override.")
    continue_parser.add_argument(
        "--print-each",
        default=3,
        type=int,
        help="Log training progress every N steps (default: 3).",
    )
    continue_parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="DataLoader workers (use 0 on Kaggle notebooks).",
    )
    continue_parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable DataParallel even if multiple GPUs are available.",
    )

    args = parser.parse_args()
    print_each = args.print_each
    num_workers = args.num_workers
    use_multi_gpu = not args.no_multi_gpu
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == "continue":
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, "options-and-config.pickle")
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, "checkpoints"))
        train_options.start_epoch = checkpoint["epoch"] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, "train")
            train_options.validation_folder = os.path.join(args.data_dir, "val")
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                raise ValueError(
                    "Requested epochs {} but checkpoint already at {}.".format(args.epochs, train_options.start_epoch)
                )
    else:
        assert args.command == "new"
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, "train"),
            validation_folder=os.path.join(args.data_dir, "val"),
            runs_folder=os.path.join(".", "runs"),
            start_epoch=1,
            experiment_name=args.name,
        )
        hidden_config = HiDDenMoEConfiguration(
            H=args.size,
            W=args.size,
            message_length=args.message,
            encoder_blocks=4,
            encoder_channels=64,
            decoder_blocks=7,
            decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3,
            discriminator_channels=64,
            decoder_loss=1.0,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=args.enable_fp16,
            num_experts=args.num_experts,
            top_k=args.top_k,
            balance_loss_weight=args.balance_loss_weight,
        )
        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, "options-and-config.pickle"), "wb+") as f:
            pickle.dump(train_options, f)
            pickle.dump([], f)
            pickle.dump(hidden_config, f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(this_run_folder, "{}.log".format(train_options.experiment_name))),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if (args.command == "new" and args.tensorboard) or (
        args.command == "continue" and os.path.isdir(os.path.join(this_run_folder, "tb-logs"))
    ):
        from tensorboard_logger import TensorBoardLogger

        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, "tb-logs"))
    else:
        tb_logger = None

    model = HiddenMoE(hidden_config, device, tb_logger)
    if use_multi_gpu:
        model.enable_multi_gpu()
    if args.command == "continue":
        logging.info("Loading checkpoint from file {}".format(loaded_checkpoint_file_name))
        model.load_from_checkpoint(checkpoint)

    logging.info("Reminder: run baseline train.py first before train_moe.py.")
    logging.info("HiDDeN MoE model: {}\n".format(model.to_stirng()))
    logging.info("Model Configuration:\n")
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info("\nTraining options:\n")
    logging.info(pprint.pformat(vars(train_options)))

    sys.stdout.flush()
    train(
        model,
        device,
        hidden_config,
        train_options,
        this_run_folder,
        tb_logger,
        print_each=print_each,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()

