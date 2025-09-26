import matplotlib.pyplot as plt
import re
import numpy as np
import argparse
import seaborn as sns
from pathlib import Path

def extract_losses_from_log(file_path):
    losses = {'clients': {}}
    with open(file_path, 'r') as file:
        for line in file:
            client_match = re.search(r"Client #(\d+).*'Round': (\d+).*'train_avg_loss': ([\d.]+)", line)
            if client_match:
                client_id = int(client_match.group(1))
                round_num = int(client_match.group(2))
                train_avg_loss = float(client_match.group(3))

                if client_id not in losses['clients']:
                    losses['clients'][client_id] = {'round': [], 'train_avg_loss': []}
                losses['clients'][client_id]['round'].append(round_num)
                losses['clients'][client_id]['train_avg_loss'].append(train_avg_loss)
    return losses

def extract_last_round_FL_training(file_path):
    last_round = None
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"Training finished in round (\d+)", line)
            if match:
                last_round = int(match.group(1))
    return last_round

def add_None_when_local_early_stop(losses, max_round):
    expected_rounds = list(range(max_round))
    for client_id, client_data in losses["clients"].items():
        rounds = client_data["round"]
        train_avg_losses = client_data["train_avg_loss"]
        if len(rounds) != len(expected_rounds):
            round_to_loss = dict(zip(rounds, train_avg_losses))
            losses_with_none = [round_to_loss.get(r, None) for r in expected_rounds]
            losses["clients"][client_id]["train_avg_loss"] = losses_with_none
            losses["clients"][client_id]["round"] = expected_rounds
    return losses

def plot_train_diagram(losses, langs, max_round, patience, save_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    clients = sorted(losses['clients'].keys())
    n_clients = len(clients)
    cmap = plt.colormaps.get_cmap('tab10').resampled(n_clients)

    for i, client in enumerate(clients):
        train_avg_loss = losses['clients'][client]['train_avg_loss']
        color = cmap(i)
        for round_idx, loss in enumerate(train_avg_loss):
            alpha = 1.0 if loss is not None else 0.15
            rect = plt.Rectangle((round_idx, i), 1, 1, facecolor=color, alpha=alpha)
            ax.add_patch(rect)

    ax.set_xlim(0, max_round)
    ax.set_ylim(0, n_clients)
    ax.set_yticks(np.arange(n_clients) + 0.5)
    ax.set_yticklabels(langs, fontsize=20)
    ticks_to_display = np.arange(0, max_round+1, 50)
    ax.set_xticks(ticks_to_display)
    ax.set_xticklabels([str(i) for i in ticks_to_display], fontsize=18)
    ax.set_xlabel('Communications Rounds', fontsize=20)
    ax.invert_yaxis()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(save_path)
    return fig

def build_influence_matrix(losses):
    clients = sorted(losses['clients'].keys())
    n_clients = len(clients)
    influence_matrix = np.zeros((n_clients, n_clients), dtype=int)

    for i, client_id in enumerate(clients):
        train_avg_loss = losses['clients'][client_id]['train_avg_loss']
        for r in range(1, len(train_avg_loss)):
            current, prev = train_avg_loss[r], train_avg_loss[r - 1]
            if current is not None and prev is None:
                for j, other_client_id in enumerate(clients):
                    if i == j: continue
                    other_loss = losses['clients'][other_client_id]['train_avg_loss']
                    if other_loss[r-1] is not None:
                        influence_matrix[i, j] += 1
    return influence_matrix, clients

def plot_influence_matrix(matrix, clients, langs, save_path='influence_matrix.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=langs, yticklabels=langs, ax=ax)
    ax.set_xlabel("Contributor Client (j)")
    ax.set_ylabel("Resuming Client (i)")
    ax.set_title("Client-to-Client Influence Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training diagram and influence matrix")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--aggregation", type=str)

    args = parser.parse_args()
    langs = ["CA", "DA", "DE", "EN", "ES", "EU", "HR", "SR"]

    log_path = f"{args.work_dir}/exp_logs/{args.model_name}/{args.dataset}_{args.c}c/{args.aggregation}.log"
    out_dir = f"{args.work_dir}/plots/{args.model_name}/{args.dataset}_{args.c}c"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    max_round = extract_last_round_FL_training(log_path)
    losses = extract_losses_from_log(log_path)
    losses = add_None_when_local_early_stop(losses, max_round)

    # Plot and save training diagram
    plot_train_diagram(losses, langs, max_round, args.patience,
                       f"{out_dir}/clients_evolution_diagram_{args.aggregation}_{args.c}c_pat{args.patience}.png")

    # Plot and save influence matrix
    influence_matrix, clients = build_influence_matrix(losses)
    plot_influence_matrix(influence_matrix, clients, langs,
                          f"{out_dir}/influence_matrix_{args.aggregation}_{args.c}c_pat{args.patience}.png")