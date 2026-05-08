# file: main.py

import argparse
import json
import os
import random

import numpy as np
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MDDD, collate_fn
from early_detection import calculate_delay
from models import LSTM_han, LSTM_attn
from utils import move_to_device

CONFIG_PATH = "./config.yaml"

class EarlyDetectionLoss(nn.Module):
    """
    A loss function that encourages early and correct predictions.
    It applies a weighted BCE loss, where the weight increases over time for positive samples
    to penalize late detections.
    """
    def __init__(self, alpha=1.0, reduction='mean'):
        """
        Args:
            alpha (float): A hyperparameter to control the penalty for lateness.
                           Higher alpha means stronger penalty.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred_seq, y_true, lengths):
        """
        Calculates the early detection loss.
        Args:
            y_pred_seq (Tensor): The model's predictions for each timestep.
                                Shape: (batch_size, max_len).
            y_true (Tensor): The ground truth labels. Shape: (batch_size, 1).
            lengths (list): A list of the true sequence lengths for each item in the batch.
        """
        if y_pred_seq.dim() != 2:
            raise ValueError(f"EarlyDetectionLoss expects 2D logits, got shape {tuple(y_pred_seq.shape)}")

        batch_size, max_len = y_pred_seq.shape
        device = y_pred_seq.device

        y_true = y_true.view(-1)
        y_true_expanded = y_true.unsqueeze(1).expand_as(y_pred_seq)

        # Calculate the base BCE loss for every timestep
        loss = self.bce_loss(y_pred_seq, y_true_expanded.float())

        # Create a mask to ignore losses from padded timesteps
        mask = torch.zeros(batch_size, max_len, device=device)
        for i, length in enumerate(lengths):
            valid_len = min(length, max_len)
            mask[i, :valid_len] = 1.0

        # Create weights to penalize late detections for positive samples
        weights = torch.ones(batch_size, max_len, device=device)
        positive_indices = y_true == 1

        if positive_indices.any():
            for i, is_positive in enumerate(positive_indices):
                if is_positive.item():
                    valid_len = min(lengths[i], max_len)
                    if valid_len > 0:
                        time_penalty = torch.linspace(0, self.alpha, valid_len, device=device)
                        weights[i, :valid_len] += time_penalty

        loss = loss * weights * mask
        
        if self.reduction == 'mean':
            return loss.sum() / mask.sum().clamp_min(1.0)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Train and test a model.")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model", type=str,
                        choices=["LSTM_han", "LSTM_attn"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_scheduler", type=str, choices=["cos", "None"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="save_dir")
    parser.add_argument("--lant_save_name", type=str, default="dep_lat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--earliness_alpha", type=float, default=1.0, help="Weight for earliness penalty in the loss function.")
    parser.set_defaults(**config)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_model_output(logits, lengths, device):
    if logits.dim() == 1:
        return logits.unsqueeze(1)
    if logits.dim() == 3 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    if logits.dim() == 2:
        batch_size = len(lengths)
        if logits.size(0) == batch_size:
            return logits

        total_steps = sum(lengths)
        if logits.size(0) == total_steps:
            max_len = max(lengths)
            aligned = torch.full(
                (batch_size, max_len),
                fill_value=-1e9,
                device=device,
                dtype=logits.dtype,
            )
            chunks = torch.split(logits, lengths, dim=0)
            for i, chunk in enumerate(chunks):
                reduced = chunk.reshape(chunk.size(0), -1).mean(dim=1)
                aligned[i, :reduced.numel()] = reduced
            return aligned

    raise RuntimeError(f"Unsupported model output shape: {tuple(logits.shape)}")


def is_sequence_output(logits):
    return logits.dim() == 2 and logits.size(1) > 1


def forward_batch(net, input_data, y, early_loss_fn, cls_loss_fn, device):
    _, user_lengths, _ = input_data
    logits = normalize_model_output(net(input_data), user_lengths, device)
    labels = y.to(torch.float32).view(-1, 1)

    if is_sequence_output(logits):
        loss = early_loss_fn(logits, labels, user_lengths)
        pred = (torch.sigmoid(logits) > 0.5).any(dim=1).long()
    else:
        logits = logits.view(-1, 1)
        loss = cls_loss_fn(logits, labels)
        pred = (torch.sigmoid(logits) > 0.5).view(-1).long()

    return loss, pred, labels.view(-1).long()


def train_epoch(net, train_loader, early_loss_fn, cls_loss_fn, optimizer, lr_scheduler, device, current_epoch, total_epochs):
    net.train()
    sample_count = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with tqdm(train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}", leave=False, unit="batch") as pbar:
        for input_data, y in pbar:
            input_data = move_to_device(input_data, device)
            y = move_to_device(y, device)

            loss, pred, labels = forward_batch(
                net, input_data, y, early_loss_fn, cls_loss_fn, device
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_size = labels.numel()
            sample_count += batch_size
            running_loss += loss.item() * batch_size
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": running_loss / sample_count})

    if lr_scheduler is not None:
        lr_scheduler.step()
        
    avg_loss = running_loss / sample_count if sample_count > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds)

    return {"loss": avg_loss, "acc": accuracy}


def val(net, val_loader, early_loss_fn, cls_loss_fn, device):
    net.eval()
    sample_count = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", leave=False, unit="batch") as pbar:
            for input_data, y in pbar:
                input_data = move_to_device(input_data, device)
                y = move_to_device(y, device)

                loss, pred, labels = forward_batch(
                    net, input_data, y, early_loss_fn, cls_loss_fn, device
                )
                
                batch_size = labels.numel()
                sample_count += batch_size
                running_loss += loss.item() * batch_size

                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                l = running_loss / sample_count
                pbar.set_postfix({"loss": l})

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / sample_count if sample_count > 0 else 0.0

    return {"loss": avg_loss, "acc": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    args = parse_args()
    seed_everything(args.seed)
    print(args)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}. "
            "Use the real dataset path, e.g. /home/gmr/tmp/hyf/MUD3_final"
        )
    train_subset = MDDD(root=args.data_dir, split='train')
    test_subset = MDDD(root=args.data_dir, split='test')
    val_subset = MDDD(root=args.data_dir, split='val')
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.model == 'LSTM_han':
        net = LSTM_han(d=256, t_downsample=4)
    elif args.model == 'LSTM_attn':
        net = LSTM_attn(d=256, t_downsample=4)
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    net = net.to(args.device)
    os.makedirs(f'./{args.save_dir}/{args.model}', exist_ok=True)


    print(f"Using EarlyDetectionLoss with alpha = {args.earliness_alpha}")
    early_loss_fn = EarlyDetectionLoss(alpha=args.earliness_alpha)
    cls_loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate)

    if args.lr_scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20
        )
    else:
        lr_scheduler = None

    best_val_loss = float('inf')
    best_model_path = f"./{args.save_dir}/{args.model}/best_model_{args.model}.pt"
    for epoch in range(args.epochs):
        train_results = train_epoch(
            net, train_loader, early_loss_fn, cls_loss_fn, optimizer, lr_scheduler,
            args.device, epoch, args.epochs
        )

        val_results = val(net, val_loader, early_loss_fn, cls_loss_fn, args.device)


        val_loss = val_results["loss"]

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)
        print('train:', train_results)
        print("val:", val_results)
    

    net.load_state_dict(
        torch.load(best_model_path, map_location=args.device)
    )

    test_results = val(net, test_loader, early_loss_fn, cls_loss_fn, args.device)
    print("Test results:", test_results)


    calculate_delay(net, test_loader, args.device, save_name=args.lant_save_name)
    print("Test early results calculated.")
    
    json_path = f"./{args.save_dir}/{args.model}/result.json"
    with open(json_path, "w") as f:
        json.dump(test_results, f)


if __name__ == '__main__':
    main()
