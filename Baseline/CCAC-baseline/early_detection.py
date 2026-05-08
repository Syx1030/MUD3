
import csv
import math
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import torch
from tqdm import tqdm

from utils import move_to_device, f_latency

cfp = 0.4545
cfn = 1
ctp = 1
ctn = 0


def lc0k(target, trajectory: list, o=50):
    time, result = 0, 0
    for i in range(0, len(trajectory)):
        if trajectory[i] == 1:
            result, time = 1, 1 - 1 / (1 + math.exp(i - o))
            break
    if result == 0 and target == 0:
        return ctn
    elif result == 1 and target == 1:
        return ctp * time
    elif result == 1 and target == 0:
        return cfp
    elif result == 0 and target == 1:
        return cfn


def batch_erde_o(batch_step_results, y, o):
    penalty = []
    for i in range(0, y.shape[0]):
        target, trajectory = y[i].item(), batch_step_results[i].reshape(-1).tolist()
        penalty.append(lc0k(target, trajectory, o))
    return penalty


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


def build_step_predictions(model, input_data, device):
    features, user_lengths, names = input_data
    logits = normalize_model_output(model(input_data), user_lengths, device)

    if logits.dim() == 2 and logits.size(1) > 1:
        batch_step_results = (torch.sigmoid(logits) > 0.5).long()
    else:
        batch_size = len(user_lengths)
        max_len = max(user_lengths)
        batch_step_results = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for current_step in range(max_len):
            prefix_features = features[:, :current_step + 1, :, :]
            prefix_lengths = [min(length, current_step + 1) for length in user_lengths]
            prefix_input = (prefix_features, prefix_lengths, names)
            prefix_logits = normalize_model_output(model(prefix_input), prefix_lengths, device)

            if prefix_logits.dim() == 2 and prefix_logits.size(1) > 1:
                step_index = torch.tensor(prefix_lengths, device=device) - 1
                step_logits = prefix_logits.gather(1, step_index.unsqueeze(1)).squeeze(1)
            else:
                step_logits = prefix_logits.view(-1)

            batch_step_results[:, current_step] = (torch.sigmoid(step_logits) > 0.5).long()

    mask = torch.zeros_like(batch_step_results)
    for i, length in enumerate(user_lengths):
        mask[i, :length] = 1
    return batch_step_results * mask


def calculate_delay(model, datasetloader, device, save_name='depression'):
    model.eval()
    all_pred, all_y, erde_5, erde_50, all_first_occurrences, delays = [], [], [], [], [], []
    with torch.no_grad():
        for input_data, y in tqdm(datasetloader, desc="Calculating Earliness"):
            input_data = move_to_device(input_data, device)
            features, user_lengths, names = input_data
            y = move_to_device(y, device).view(-1)

            batch_step_results = build_step_predictions(model, input_data, device)
            first_occurrences = []
            for i in range(batch_step_results.shape[0]):
                trace = batch_step_results[i, :user_lengths[i]].cpu().tolist()
                if y[i].item() == 1:
                    first_step = (batch_step_results[i] == 1).nonzero(as_tuple=True)[0]
                    if len(first_step) > 0:
                        first_occurrences.append((first_step[0].item(), names[i], trace))
                        delays.append(first_step[0].item())
                    else:
                        first_occurrences.append(('None', names[i], trace))
                        delays.append(user_lengths[i])
                else:
                    delays.append(0)
            all_first_occurrences = all_first_occurrences + first_occurrences
            erde_5 = erde_5 + batch_erde_o(batch_step_results, y, 5)
            erde_50 = erde_50 + batch_erde_o(batch_step_results, y, 10)
            y_pred = (torch.sum(batch_step_results, dim=1) > 0).long()
            all_y.append(y.reshape(-1).cpu())
            all_pred.append(y_pred.reshape(-1).cpu())

    # Convert lists of tensors to single tensors
    all_y = torch.cat(all_y)
    all_pred = torch.cat(all_pred)

    # Calculate metrics
    f1 = f1_score(all_y.numpy(), all_pred.numpy(), average='binary', zero_division=0)
    accuracy = accuracy_score(all_y.numpy(), all_pred.numpy())
    recall = recall_score(all_y.numpy(), all_pred.numpy(), average='binary', zero_division=0)
    precision = precision_score(all_y.numpy(), all_pred.numpy(), average='binary', zero_division=0)
    erde_5 = sum(erde_5) / len(erde_5) if erde_5 else 0.0
    erde_50 = sum(erde_50) / len(erde_50) if erde_50 else 0.0
    f1_latency = f_latency(all_pred.numpy(), all_y.numpy(), np.array(delays))
    delay_data = {'delay': [f[0] for f in all_first_occurrences], 'name': [f[1] for f in all_first_occurrences],
                  'trace': [f[2] for f in all_first_occurrences]}
    file_name = f'latency_{save_name}.csv'
    is_new_file = not os.path.exists(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['delay', 'name', 'trace'])
        if is_new_file:
            writer.writeheader()
        for i in range(len(delay_data['delay'])):
            writer.writerow({
                'delay': delay_data['delay'][i],
                'name': delay_data['name'][i],
                'trace': str(delay_data['trace'][i])
            })
    print(f"数据已成功写入 '{file_name}' 文件")
    print(f"F1 Score: {f1}")
    print(f"F1_latency: {f1_latency}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"ERDE5: {erde_5}")
    print(f"ERDE10: {erde_50}")
