import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.ADCRezero.data.onthefly import RezeroOnTheFlyDataset
from src.ADCRezero.data.freeze import RezeroFreezeDataset, collate_fn
from src.ADCRezero.models.ARezero import ARezeroModel
from src.ADCRezero.models.CRezero import CRezeroModel
from src.ADCRezero.models.DRezero import DRezeroModel
from tqdm import tqdm
from src.utils.trainlogger import TrainLogger
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import traceback
import soundfile as sf
import numpy as np

def compute_freq_mae_loss(x_hat_spec: torch.Tensor, weight: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Frequency-domain MAE loss on complex spectrogram.
    L = weight * (|Re(Z)| + |Im(Z)|)_mean
    x_hat_spec: [B, F, T]
    return: [B,]
    """
    mae = weight * (x_hat_spec.real.abs().sum(dim=(1, 2)) + x_hat_spec.imag.abs().sum(dim=(1, 2)))
    return mae

def compute_snr_loss(y: torch.Tensor, y_hat: torch.Tensor, 
                    snr_max_db: float = 30.0, eps: float = 1e-8, weight: float = 1.0) -> torch.Tensor:
    """
    クリップ付きSNR損失
    L = 10*log10((||e||^2 + tau*||y||^2) / (||y||^2)),  tau = 10^(-SNR_max/10)
    y: [B, T]
    y_hat: [B, T]
    return: [B,]
    """
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {y_hat.shape}")

    err = y - y_hat
    power_e = (err ** 2).sum(dim=1)
    power_y = (y ** 2).sum(dim=1)
    tau = 10.0 ** (-snr_max_db / 10.0)
    loss = 10.0 * torch.log10(power_e + tau * power_y)
    loss = weight * loss
    return loss

def nan_hook(module, input, output):
    """
    モジュールの出力 (Tensor, tuple, list, dict) を再帰的にチェックし、
    NaN を含むテンソルがあればモジュール名と出力パスを出力して例外を投げる。
    """
    def check(x, path="output"):
        if torch.is_tensor(x):
            if torch.isnan(x).any():
                print(f"[NaN Detected] Module: {module.__class__.__name__}, Path: {path}, Shape: {tuple(x.shape)}")
                traceback.print_stack()
                raise RuntimeError(f"NaN in {module.__class__.__name__} at {path}")
        elif isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                check(xi, f"{path}[{i}]")
        elif isinstance(x, dict):
            for k, xi in x.items():
                check(xi, f"{path}['{k}']")
    check(output)


def main():
    parser = argparse.ArgumentParser(description="Train ReZero model with WandB logging")
    parser.add_argument('--speech_dir', required=True, help='Path to directory with clean speech wavs')
    parser.add_argument('--noise_dir', required=True, help='Path to directory with noise wavs')
    parser.add_argument('--config', default='src/ADCRezero/config/original.yaml', help='Path to config.yaml')
    parser.add_argument('--device', default='cuda', help='Device to use for training')
    parser.add_argument('--val_dir', required=True, help='Path to validation dataset directory')
    parser.add_argument('--output_dir', default='models', help='Directory to save model checkpoints')
    parser.add_argument('--val_audio_dir', default='outputs/val_audio', help='Directory to save validation audio samples')
    parser.add_argument("--mic_arch", choices=["circular", "linear"], default="circular", help="Microphone array architecture")
    parser.add_argument("--region_type", choices=["angular", "spherical", "conical"], default="conical", help="Type of query region to sample")
    parser.add_argument("--first_positioning", choices=["mic", "speaker"], default="mic", help="First positioning of sources")
    parser.add_argument("--decision_query_region", choices=['no_limit', 'angle_limit', 'distance_limit'], default="no_limit", help="Method for deciding query region if args.first_positioning is 'mic'")
    parser.add_argument("--limit_mic_z", action='store_true', help="Limit microphone z-coordinate to a specific range")
    parser.add_argument("--elevation_limit", action='store_true', help="Limit elevation of sources to a specific range")
    parser.add_argument("--mic_in_center", action='store_true', help="Place the microphone array at the center of the room")
    parser.add_argument("--room_size", type=float, nargs=3, default=[6.0, 6.0, 3.5], help="Room size in meters (x y z)")
    parser.add_argument("--infinity_room", action='store_true', help="Use an infinite room for simulation (no reverberation)")
    parser.add_argument("--query2D", action='store_true', help="Consider the query area in a two-dimensional plane.")
    parser.add_argument("--spherical_fixed_azimuth", action='store_true', help="For spherical region type, fix azimuth angles when sampling sources.")
    parser.add_argument("--rms_norm", action="store_true", help="Apply RMS normalization to the mixture")
    parser.add_argument("--no_noise", action='store_true', help="Exclude noise sources from the mixtures")
    parser.add_argument("--diffuse_noise_prob", type=float, default=0.1, help="Probability of using diffuse noise when n_n > 0")
    parser.add_argument("--one_speaker", action='store_true', help="Use only one speaker in the simulation")
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--project_name', required=True, help='WandB project name')
    parser.add_argument('--run_name', default=None, help='WandB run name')
    parser.add_argument('--run_id', default=None, help='WandB run ID')
    parser.add_argument('--resume_checkpoint_path', help='Resume checkpoint path')
    parser.add_argument("--cpuram", action='store_true', help="Load audio files into RAM")
    args = parser.parse_args()
    args.mode = "train"
    args.dataset_generation = False

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    training_cfg = cfg['training']
    batch_size = training_cfg['batch_size']
    num_workers = training_cfg['num_workers']
    max_iters = training_cfg['iterations']
    iterations_per_epoch = training_cfg['iterations_per_epoch']
    opt_cfg = training_cfg['optimizer']
    initial_lr = opt_cfg['initial_lr']
    decay_cfg = opt_cfg['lr_decay']
    decay_factor = decay_cfg['factor']
    decay_epochs = decay_cfg['every_epochs']

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build model
    if args.region_type == "angular":
        model = ARezeroModel(
            args,
            device=device,
            return_mask=False,
            complex_as_channel=True
        )
    elif args.region_type == "spherical":
        model = DRezeroModel(
            args,
            device=device,
            return_mask=False,
            complex_as_channel=True
        )
    elif args.region_type == "conical":
        model = CRezeroModel(
            args,
            device=device,
            return_mask=False,
            complex_as_channel=True
        )
    else:
        raise ValueError(f"Unsupported region type: {args.region_type}")
    
    now = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d-%H:%M:%S")
    if args.run_name is None:
        run_name = now
    else:
        run_name = f"{args.run_name}_{now}"
    output_dir = os.path.join(args.output_dir, args.project_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_yaml_path = os.path.join(output_dir, "log.yaml")
    
    iteration = 0
    if args.resume_checkpoint_path:
        state = torch.load(args.resume_checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model = model.float().to(device)
        iteration = int(args.resume_checkpoint_path.split('_')[-1].split('.')[0])
        if not args.onlyQtrain:
            initial_lr = initial_lr * (decay_factor ** (iteration // (decay_epochs * iterations_per_epoch)))
            print(f"Resumed model from {args.resume_checkpoint_path} at iteration {iteration}, adjusted initial LR to {initial_lr}")
        else:
            initial_lr = initial_lr
            print(f"Resumed model from {args.resume_checkpoint_path} at iteration {iteration} for Q-only training")
        # Setup WandB logger with resume
        logger = TrainLogger(
            use_wandb=not args.no_wandb,
            project_name=args.project_name,
            run_name=run_name,
            run_id=args.run_id,
            log_yaml_path=log_yaml_path,
            config=cfg,
        )
    else:
        # Setup WandB logger
        logger = TrainLogger(
            use_wandb=not args.no_wandb,
            project_name=args.project_name,
            run_name=run_name,
            log_yaml_path=log_yaml_path,
            config=cfg,
        )
    logger.start()
    model.to(device)
    
    # Debugging hook to detect NaNs
    # for m in model.modules():
    #     m.register_forward_hook(nan_hook)

    # Prepare dataset and dataloader
    train_dataset = RezeroOnTheFlyDataset(args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    train_data_iter = iter(train_dataloader)
    
    val_dataset = RezeroFreezeDataset(args, args.val_dir)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_fn
    )
    val_data_iter = iter(val_dataloader)
    
    # STFT, iSTFT parameters from dataset
    n_fft = train_dataset.n_fft
    hop_length=train_dataset.hop
    window = torch.hann_window(n_fft).to(device) if train_dataset.win_type=='hann' else torch.ones(n_fft).to(device)
    print(f"STFT parameters: n_fft={n_fft}, hop_length={hop_length}, window_type={train_dataset.win_type}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=decay_epochs,
        gamma=decay_factor
    )

    # Loss weight for Q==0
    lambda_q0 = training_cfg['loss']['Q_eq_0']['lambda']
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params}, trainable={trainable_params}")

    # Training loop
    num_epochs = math.ceil(max_iters / iterations_per_epoch)
    for epoch in tqdm(range(1, num_epochs + 1)):
        # --- TRAINING ---
        for _ in tqdm(range(iterations_per_epoch)):
            if iteration >= max_iters:
                break
            iteration += 1

            try:
                batch = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_dataloader)
                batch = next(train_data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            mix = batch["mix"]
            Q = batch["Q"]
            target = batch["target"]                  # [B, C, T]
            target = target[:, 0:1, :].squeeze(1)     # [B, T]

            if args.region_type == "angular":
                theta_l = batch["theta_l"]
                theta_h = batch["theta_h"]
                x_hat_spec = model(mix, theta_l, theta_h).squeeze(1)     # [B, F, T_spec]

            elif args.region_type == "spherical":
                d_query = batch["d_query"]
                x_hat_spec = model(mix, d_query).squeeze(1)              # [B, F, T_spec]

            elif args.region_type == "conical":
                theta_l = batch["theta_l"]
                theta_h = batch["theta_h"]
                d_query = batch["d_query"]
                x_hat_spec = model(mix, theta_l, theta_h, d_query).squeeze(1)  # [B, F, T_spec]

            mask_q0 = (Q == 0)       # (B,) bool
            mask_snr = ~mask_q0

            # batched iSTFT for all samples
            x_hat_time = torch.istft(
                x_hat_spec,                      # (B, F, T_spec) complex
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                length=mix.size(-1),
            )                                    # -> (B, T)

            # per-sample losses (B,)
            train_mae_loss_per = compute_freq_mae_loss(x_hat_spec, lambda_q0)  # (B,)
            train_snr_loss_per = compute_snr_loss(target, x_hat_time)          # (B,)

            train_loss_per = torch.where(mask_q0, train_mae_loss_per, train_snr_loss_per)  # (B,)
            train_loss = train_loss_per.mean()

            # masked means for logging
            train_mae_loss = train_mae_loss_per[mask_q0].mean() if mask_q0.any() else None
            train_snr_loss = train_snr_loss_per[mask_snr].mean() if mask_snr.any() else None

            # Backward and update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            print(f"==========Iter {iteration}/{max_iters} - Train Loss: {train_loss.item():.4f}==========")

            # Log training loss and learning rate (wandb/logger)
            logs = {
                "Iteration": iteration,
                "Learning Rate": scheduler.get_last_lr()[0],
                "Train Loss": train_loss.detach().item(),
            }
            if train_mae_loss is not None:
                logs["Train MAE Loss"] = train_mae_loss.detach().item()
            if train_snr_loss is not None:
                logs["Train SNR Loss"] = train_snr_loss.detach().item()
            logger.log(logs)

            # Checkpointing every 10000 iterations
            if iteration == 1000 or iteration == 2000 or iteration == 5000 or iteration % 10000 == 0:
                ckpt_path = os.path.join(output_dir, f"model_iter_{iteration}.pth")
                torch.save(model.state_dict(), ckpt_path)

            # Validation
            if (
                iteration == 1
                or (iteration <= 1000 and (iteration == 100 or iteration % 500 == 0))
                or (iteration > 1000 and (iteration == 2000 or iteration == 5000 or iteration % 10000 == 0))
            ):
                model.eval()
                val_losses = []
                val_mae_losses = []
                val_snr_losses = []
                save_audio = True

                with torch.no_grad():
                    for val_batch in tqdm(val_dataloader):
                        new_val = {}
                        for k, v in val_batch.items():
                            if isinstance(v, (tuple, list)):
                                new_val[k] = type(v)(item.to(device) for item in v)
                            else:
                                new_val[k] = v.to(device)
                        val_batch = new_val

                        mix = val_batch["mix"]
                        Q = val_batch["Q"]
                        target = val_batch["target"]                  # [B, C, T]
                        target = target[:, 0:1, :].squeeze(1)         # [B, T]

                        if args.region_type == "angular":
                            theta_l = val_batch["theta_l"]
                            theta_h = val_batch["theta_h"]
                            x_hat_spec = model(mix, theta_l, theta_h).squeeze(1)

                        elif args.region_type == "spherical":
                            d_query = val_batch["d_query"]
                            x_hat_spec = model(mix, d_query).squeeze(1)

                        elif args.region_type == "conical":
                            theta_l = val_batch["theta_l"]
                            theta_h = val_batch["theta_h"]
                            d_query = val_batch["d_query"]
                            x_hat_spec = model(mix, theta_l, theta_h, d_query).squeeze(1)

                        # Save audio from the first validation batch (ここはログ用途なので for は残しています)
                        if save_audio:
                            mix_list = []
                            target_list = []
                            est_list = []
                            Q_list = Q.cpu().numpy().tolist()

                            for i in range(val_batch["mix"].size(0)):
                                mix_i = val_batch["mix"][i, 0, :]
                                target_i = target[i]
                                x_hat_i = torch.istft(
                                    x_hat_spec[i],
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    window=window,
                                    length=val_batch["mix"].size(-1),
                                )
                                mix_list.append(mix_i.cpu().numpy())
                                target_list.append(target_i.cpu().numpy())
                                est_list.append(x_hat_i.cpu().numpy())

                            for idx, (mix_i, target_i, est_i, Q_i) in enumerate(zip(mix_list, target_list, est_list, Q_list)):
                                logger.log(
                                    data={
                                        f"val/mix_{idx}_Q={Q_i}": mix_i.astype(np.float32),
                                        f"val/target_{idx}_Q={Q_i}": target_i.astype(np.float32),
                                        f"val/est_{idx}_Q={Q_i}": est_i.astype(np.float32),
                                    },
                                    option="audio",
                                    option_config={
                                        "sample_rate": train_dataset.sr,
                                        "caption": f"iter{iteration}_sample{idx}",
                                    },
                                )
                            save_audio = False

                        mask_q0 = (Q == 0)
                        mask_snr = ~mask_q0

                        x_hat_time = torch.istft(
                            x_hat_spec,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=mix.size(-1),
                        )

                        val_mae_loss_per = compute_freq_mae_loss(x_hat_spec, lambda_q0)  # (B,)
                        val_snr_loss_per = compute_snr_loss(target, x_hat_time)          # (B,)

                        val_loss_per = torch.where(mask_q0, val_mae_loss_per, val_snr_loss_per)
                        val_losses.append(val_loss_per.mean())

                        if mask_q0.any():
                            val_mae_losses.append(val_mae_loss_per[mask_q0].mean())
                        if mask_snr.any():
                            val_snr_losses.append(val_snr_loss_per[mask_snr].mean())

                avg_val_loss = torch.stack(val_losses).mean().item()
                logger.log({"Validation Loss": avg_val_loss})

                if len(val_mae_losses) != 0:
                    avg_val_mae_loss = torch.stack(val_mae_losses).mean().item()
                    logger.log({"Validation MAE Loss": avg_val_mae_loss})

                if len(val_snr_losses) != 0:
                    avg_val_snr_loss = torch.stack(val_snr_losses).mean().item()
                    logger.log({"Validation SNR Loss": avg_val_snr_loss})

                print(f"---------- Iteration {iteration} Validation Loss: {avg_val_loss:.4f} ----------")
                model.train()

        # Step LR scheduler at end of epoch
        scheduler.step()
        if iteration > max_iters:
            break

    # Finishing up
    logger.finish()


if __name__ == '__main__':
    main()