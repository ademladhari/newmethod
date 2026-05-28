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
    save_every: int = 1,
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
                if optim is None:
                    continue
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

        if epoch >= 20 and training_losses["expert_max_sel "].avg > 0.8:
            logging.warning(
                "Expert usage warning: max expert selection rate is {:.4f} (> 0.80).".format(
                    training_losses["expert_max_sel "].avg
                )
            )

        train_duration = time.time() - epoch_start
        logging.info("Epoch {} training duration {:.2f} sec".format(epoch, train_duration))
        logging.info("-" * 40)
        utils.write_losses(os.path.join(this_run_folder, "train.csv"), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        val_batches = 0
        validation_losses = defaultdict(AverageMeter)
        logging.info("Running validation for epoch {}/{}".format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            val_batches += 1
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages, router_probs, topk_indices, router_logits) = model.validate_on_batch([image, message])
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

        if val_batches > 0:
            log_progress_and_flush(validation_losses)
        else:
            logging.warning(
                "Validation skipped for epoch {} because no validation batches were found in {}.".format(
                    epoch, train_options.validation_folder
                )
            )
        logging.info("-" * 40)
        sys.stdout.flush()
        should_save_checkpoint = (epoch % save_every == 0) or (epoch == train_options.number_of_epochs)
        if should_save_checkpoint:
            model.save_checkpoint(
                train_options.experiment_name,
                epoch,
                os.path.join(this_run_folder, "checkpoints"),
            )
        if val_batches > 0:
            utils.write_losses(
                os.path.join(this_run_folder, "validation.csv"),
                validation_losses,
                epoch,
                time.time() - epoch_start,
            )


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
    new_run_parser.add_argument(
        "--balance-loss-start-weight",
        default=0.01,
        type=float,
        help="Initial balance weight used during warmup.",
    )
    new_run_parser.add_argument(
        "--balance-loss-warmup-epochs",
        default=20,
        type=int,
        help="Epochs used to linearly warm balance loss weight.",
    )
    new_run_parser.add_argument("--router-jitter-noise", default=0.01, type=float, help="Router jitter noise.")
    new_run_parser.add_argument("--router-input-dropout", default=0.1, type=float, help="Router input dropout.")
    new_run_parser.add_argument("--router-z-loss-weight", default=0.001, type=float, help="Router z-loss weight.")
    new_run_parser.add_argument("--router-temperature-start", default=1.0, type=float, help="Initial router temperature.")
    new_run_parser.add_argument("--router-temperature-end", default=1.0, type=float, help="Final router temperature.")
    new_run_parser.add_argument(
        "--router-grad-clip-norm",
        default=1.0,
        type=float,
        help="Clip norm threshold for router gradients (0 disables).",
    )
    new_run_parser.add_argument("--expert-dropout", default=0.1, type=float, help="Dropout used inside each expert.")
    new_run_parser.add_argument(
        "--expert-weight-decay",
        default=1e-4,
        type=float,
        help="Weight decay applied to expert parameters only.",
    )
    new_run_parser.add_argument(
        "--expert-init-offset-scale",
        default=1e-3,
        type=float,
        help="Scale of per-expert random init offset for symmetry breaking.",
    )
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
    new_run_parser.add_argument(
        "--save-every",
        default=1,
        type=int,
        help="Save checkpoint every N epochs (default: 1).",
    )
    new_run_parser.add_argument(
        "--init-hidden-checkpoint",
        default="",
        type=str,
        help="Optional baseline HiDDeN checkpoint (.pyt) used to warm-start encoder/backbone.",
    )
    new_run_parser.add_argument(
        "--freeze-hidden-backbone",
        action="store_true",
        help="Freeze encoder and decoder feature backbone; train MoE-specific parts only.",
    )
    new_run_parser.add_argument(
        "--freeze-discriminator",
        action="store_true",
        help="Freeze discriminator weights.",
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
    continue_parser.add_argument(
        "--save-every",
        default=1,
        type=int,
        help="Save checkpoint every N epochs (default: 1).",
    )
    continue_parser.add_argument(
        "--freeze-hidden-backbone",
        default=None,
        choices=["true", "false"],
        type=str,
        help="Optional override for freeze_hidden_backbone from stored config.",
    )
    continue_parser.add_argument(
        "--freeze-discriminator",
        default=None,
        choices=["true", "false"],
        type=str,
        help="Optional override for freeze_discriminator from stored config.",
    )

    args = parser.parse_args()
    print_each = args.print_each
    num_workers = args.num_workers
    save_every = max(1, args.save_every)
    use_multi_gpu = not args.no_multi_gpu
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == "continue":
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, "options-and-config.pickle")
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        for key, default_value in (
            ("router_jitter_noise", 0.01),
            ("router_input_dropout", 0.1),
            ("router_z_loss_weight", 0.001),
            ("router_temperature_start", 1.0),
            ("router_temperature_end", 1.0),
            ("router_grad_clip_norm", 1.0),
            ("balance_loss_start_weight", 0.01),
            ("balance_loss_warmup_epochs", 20),
            ("expert_dropout", 0.1),
            ("expert_weight_decay", 1e-4),
            ("expert_init_offset_scale", 1e-3),
            ("init_hidden_checkpoint", ""),
            ("freeze_hidden_backbone", False),
            ("freeze_discriminator", False),
        ):
            if not hasattr(hidden_config, key):
                setattr(hidden_config, key, default_value)
        if args.freeze_hidden_backbone is not None:
            hidden_config.freeze_hidden_backbone = args.freeze_hidden_backbone.lower() == "true"
        if args.freeze_discriminator is not None:
            hidden_config.freeze_discriminator = args.freeze_discriminator.lower() == "true"
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
            balance_loss_start_weight=args.balance_loss_start_weight,
            balance_loss_warmup_epochs=args.balance_loss_warmup_epochs,
            router_jitter_noise=args.router_jitter_noise,
            router_input_dropout=args.router_input_dropout,
            router_z_loss_weight=args.router_z_loss_weight,
            router_temperature_start=args.router_temperature_start,
            router_temperature_end=args.router_temperature_end,
            router_grad_clip_norm=args.router_grad_clip_norm,
            expert_dropout=args.expert_dropout,
            expert_weight_decay=args.expert_weight_decay,
            expert_init_offset_scale=args.expert_init_offset_scale,
            init_hidden_checkpoint=args.init_hidden_checkpoint,
            freeze_hidden_backbone=args.freeze_hidden_backbone,
            freeze_discriminator=args.freeze_discriminator,
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
    if args.command == "new" and hidden_config.init_hidden_checkpoint:
        hidden_checkpoint_path = hidden_config.init_hidden_checkpoint
        if not os.path.isfile(hidden_checkpoint_path):
            raise FileNotFoundError("Hidden checkpoint file not found: {}".format(hidden_checkpoint_path))
        logging.info("Initializing from baseline checkpoint {}".format(hidden_checkpoint_path))
        warmstart_checkpoint = torch.load(hidden_checkpoint_path, map_location=device)
        model.initialize_from_hidden_checkpoint(
            warmstart_checkpoint,
            load_discriminator=not bool(getattr(hidden_config, "freeze_discriminator", False)),
        )
    model.apply_freeze_settings()
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

    hidden_config.number_of_epochs = train_options.number_of_epochs
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
        save_every=save_every,
    )


if __name__ == "__main__":
    main()

