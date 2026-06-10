#!/usr/bin/env python3
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import FSPNet_model
import dataset
import loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("FSPNet-Transformer Single-GPU Training")

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--path', type=str, default='./Datasets', help='train dataset path')

    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--dataset', type=str, default='ACOD-12K', help='path to train dataset')

    parser.add_argument('--base-lr', default=1e-4, type=float)
    parser.add_argument('--save-dir', default='./Checkpoints/FSPNet', type=str)
    parser.add_argument('--save-epoch', default=2, type=int)

    parser.add_argument('--pretrain', type=str, default=None, help='load encoder pretrain')
    parser.add_argument('--resume', type=str, default=None, help='resume full checkpoint')
    parser.add_argument('--ft_for_MoCA', type=str, default=None)

    parser.add_argument('--use-amp', action='store_true', help='use torch.cuda.amp')

    return parser.parse_args()


# ---------------------------------------------------------------------------
#  ** Single GPU Training Only **
# ---------------------------------------------------------------------------

def build_model(args):
    torch.cuda.set_device(args.gpu)

    net = FSPNet_model.Model(args.pretrain, img_size=704)
    net = net.cuda(args.gpu)

    model_without_ddp = net  # for API compatibility but no DDP

    return net, model_without_ddp


def build_dataloader(args):
    Dataset = dataset.TrainDataset([args.path])

    Dataloader = DataLoader(
        Dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.my_collate_fn,
        drop_last=True
    )
    return Dataset, Dataloader


def load_checkpoint_if_present(path, model, optimizer=None, device='cpu'):
    if path is None or not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)

    # Try to load model weights
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
        print(f"Loaded checkpoint weights from {path}")
    except Exception:
        print("Strict load failed — loading with strict=False")
        model.load_state_dict(state, strict=False)

    # Try to load optimizer
    if optimizer is not None and isinstance(ckpt, dict) and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("Loaded optimizer state.")
        except Exception as e:
            print("Could not load optimizer:", e)

    return ckpt


# ---------------------------------------------------------------------------
#  TRAIN
# ---------------------------------------------------------------------------

def main(args):

    # ------------------------- MODEL -------------------------
    net, model = build_model(args)

    # ------------------------ OPTIMIZER -----------------------
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.base_lr * 0.1},
        {"params": decoder_params, "lr": args.base_lr}
    ])

    # -------------------------- LOAD --------------------------
    device = f"cuda:{args.gpu}"

    if args.resume:
        load_checkpoint_if_present(args.resume, model, optimizer, device)

    if args.ft_for_MoCA:
        ckpt = torch.load(args.ft_for_MoCA, map_location=device)
        try:
            model.load_state_dict(ckpt)
        except:
            model.load_state_dict(ckpt, strict=False)
        print("Fine-tuning from MoCA ckpt:", args.ft_for_MoCA)

    # -------------------------- DATA --------------------------
    Dataset, Dataloader = build_dataloader(args)

    torch.backends.cudnn.benchmark = True

    # AMP
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    start_time = time.time()

    # ------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------
    for epoch in range(args.num_epochs):

        # Manual LR drop at epoch 49 (same as original)
        if epoch == 49:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1
                print("LR decayed to:", pg['lr'])

        model.train()

        loss_all_sum = 0
        loss_main_sum = 0
        count = 0

        for batch in Dataloader:
            count += 1

            img = batch['img'].cuda(args.gpu, non_blocking=True)
            label = batch['label'].cuda(args.gpu, non_blocking=True)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(img)
                    all_loss, m_loss = loss.multi_bce(out, label)
                scaler.scale(all_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(img)
                all_loss, m_loss = loss.multi_bce(out, label)
                all_loss.backward()
                optimizer.step()

            loss_all_sum += all_loss.item()
            loss_main_sum += m_loss.item()

            if count % 20 == 0:
                print(f"[Epoch {epoch:03d}] Iter {count:04d} | "
                      f"Loss_all: {loss_all_sum/count:.5f} | Loss_main: {loss_main_sum/count:.5f}")

        if epoch % args.save_epoch == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(
                args.save_dir,
                f"fspnet_epoch{epoch}_mainloss{loss_main_sum/count:.5f}.pth"
            )
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, save_path)

            print("Saved:", save_path)

    elapsed = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed:.2f} minutes.")


if __name__ == '__main__':
    args = parse_args()

    args.save_dir = os.path.join(args.save_dir, args.dataset)
    args.path = os.path.join(args.path, args.dataset, "Train")
    
    main(args)
