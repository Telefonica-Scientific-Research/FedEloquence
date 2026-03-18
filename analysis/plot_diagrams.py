import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLIENT_LOSS_PATTERN = re.compile(
    r"Client #(\d+).*'Round': (\d+).*'train_avg_loss': ([\d.]+)"
)
LAST_ROUND_PATTERN = re.compile(r"Training finished in round (\d+)")
CLIENTS_PATTERN = re.compile(r"(\d+)c$", re.IGNORECASE)
PATIENCE_PAT_PREFIX_PATTERN = re.compile(r"pat(\d+)$", re.IGNORECASE)
PATIENCE_PAT_SUFFIX_PATTERN = re.compile(r"(\d+)pat$", re.IGNORECASE)
MONO_PATTERN = re.compile(r"mono\d+$", re.IGNORECASE)
MULTI_PATTERN = re.compile(r"multi\d+$", re.IGNORECASE)
MONO100_BASE_LANGUAGE_ORDER = ["CA", "DA", "DE", "EN", "ES", "EU", "HR", "SR"]
MONO100_TARGET_LANGUAGE_ORDER = ["EN", "ES", "DE", "CA", "DA", "SR", "HR", "EU"]

def extract_losses_from_log(file_path):
    losses = {"clients": {}}
    with open(file_path, "r") as file:
        for line in file:
            client_match = CLIENT_LOSS_PATTERN.search(line)
            if client_match:
                client_id = int(client_match.group(1))
                round_num = int(client_match.group(2))
                train_avg_loss = float(client_match.group(3))

                if client_id not in losses["clients"]:
                    losses["clients"][client_id] = {"round": [], "train_avg_loss": []}
                losses["clients"][client_id]["round"].append(round_num)
                losses["clients"][client_id]["train_avg_loss"].append(train_avg_loss)
    return losses


def extract_last_round_fl_training(file_path):
    last_round = None
    with open(file_path, "r") as file:
        for line in file:
            match = LAST_ROUND_PATTERN.search(line)
            if match:
                last_round = int(match.group(1))
    return last_round


def infer_num_rounds(losses, last_round):
    observed_rounds = [
        max(client_data["round"]) + 1
        for client_data in losses["clients"].values()
        if client_data["round"]
    ]
    observed_num_rounds = max(observed_rounds, default=0)
    if last_round is None:
        return observed_num_rounds
    return max(last_round, observed_num_rounds)


def add_none_when_local_early_stop(losses, num_rounds):
    expected_rounds = list(range(num_rounds))
    for client_id, client_data in losses["clients"].items():
        rounds = client_data["round"]
        train_avg_losses = client_data["train_avg_loss"]
        if len(rounds) != len(expected_rounds):
            round_to_loss = dict(zip(rounds, train_avg_losses))
            losses_with_none = [round_to_loss.get(r, None) for r in expected_rounds]
            losses["clients"][client_id]["train_avg_loss"] = losses_with_none
            losses["clients"][client_id]["round"] = expected_rounds
    return losses


def get_row_order_and_labels(client_ids, composition):
    sorted_clients = sorted(client_ids)
    if (
        normalize(composition) == "mono100_multi0"
        and len(sorted_clients) == len(MONO100_BASE_LANGUAGE_ORDER)
    ):
        client_by_language = {
            language: sorted_clients[idx]
            for idx, language in enumerate(MONO100_BASE_LANGUAGE_ORDER)
        }
        ordered_clients = [client_by_language[language] for language in MONO100_TARGET_LANGUAGE_ORDER]
        return ordered_clients, MONO100_TARGET_LANGUAGE_ORDER

    labels = [f"C{i}" for i in range(1, len(sorted_clients) + 1)]
    return sorted_clients, labels


def plot_train_diagram(losses, client_order, labels, num_rounds, save_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    n_clients = len(client_order)
    cmap = plt.colormaps.get_cmap('tab10').resampled(n_clients)

    for i, client_id in enumerate(client_order):
        train_avg_loss = losses["clients"][client_id]["train_avg_loss"]
        color = cmap(i)
        for round_idx, loss in enumerate(train_avg_loss):
            alpha = 1.0 if loss is not None else 0.15
            rect = plt.Rectangle((round_idx, i), 1, 1, facecolor=color, alpha=alpha)
            ax.add_patch(rect)

    ax.set_xlim(0, num_rounds)
    ax.set_ylim(0, n_clients)
    ax.set_yticks(np.arange(n_clients) + 0.5)
    ax.set_yticklabels(labels, fontsize=20)
    ticks_to_display = np.arange(0, num_rounds + 1, 50)
    ax.set_xticks(ticks_to_display)
    ax.set_xticklabels([str(i) for i in ticks_to_display], fontsize=18)
    ax.set_xlabel("Communications Rounds", fontsize=20)
    ax.invert_yaxis()
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def build_influence_matrix(losses, client_order):
    n_clients = len(client_order)
    influence_matrix = np.zeros((n_clients, n_clients), dtype=int)

    for i, client_id in enumerate(client_order):
        train_avg_loss = losses["clients"][client_id]["train_avg_loss"]
        for r in range(1, len(train_avg_loss)):
            current, prev = train_avg_loss[r], train_avg_loss[r - 1]
            if current is not None and prev is None:
                for j, other_client_id in enumerate(client_order):
                    if i == j:
                        continue
                    other_loss = losses["clients"][other_client_id]["train_avg_loss"]
                    if other_loss[r - 1] is not None:
                        influence_matrix[i, j] += 1
    return influence_matrix


def plot_influence_matrix(matrix, langs, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=langs, yticklabels=langs, ax=ax)
    ax.set_xlabel("Contributor Client (j)")
    ax.set_ylabel("Resuming Client (i)")
    ax.set_title("Client-to-Client Influence Matrix")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def parse_log_metadata(log_path):
    stem = log_path.stem
    if stem.startswith("exp_print_"):
        stem = stem[len("exp_print_") :]
    tokens = stem.split("_")

    metadata = {
        "clients": None,
        "patience": None,
        "early_stop": None,
        "fl_method": None,
        "composition": None,
    }

    for token in tokens:
        match = CLIENTS_PATTERN.fullmatch(token)
        if match:
            metadata["clients"] = int(match.group(1))
            break

    for token in tokens:
        match_prefix = PATIENCE_PAT_PREFIX_PATTERN.fullmatch(token)
        if match_prefix:
            metadata["patience"] = int(match_prefix.group(1))
            break
        match_suffix = PATIENCE_PAT_SUFFIX_PATTERN.fullmatch(token)
        if match_suffix:
            metadata["patience"] = int(match_suffix.group(1))
            break

    composition_index = None
    for idx in range(len(tokens) - 1):
        if MONO_PATTERN.fullmatch(tokens[idx]) and MULTI_PATTERN.fullmatch(tokens[idx + 1]):
            metadata["composition"] = f"{tokens[idx]}_{tokens[idx + 1]}"
            composition_index = idx
            break

    if composition_index is not None and composition_index >= 1:
        metadata["fl_method"] = tokens[composition_index - 1]

    if composition_index is not None and composition_index >= 2:
        early_stop_candidate = tokens[composition_index - 2]
        if early_stop_candidate.isalpha():
            metadata["early_stop"] = early_stop_candidate

    return metadata


def normalize(text):
    return text.lower()


def log_matches_filters(metadata, args):
    if metadata["clients"] is not None and metadata["clients"] != args.c:
        return False

    if metadata["patience"] is not None and metadata["patience"] != args.patience:
        return False

    if metadata["fl_method"] is not None:
        if normalize(metadata["fl_method"]) != normalize(args.fl_method):
            return False

    if metadata["early_stop"] is not None:
        if normalize(metadata["early_stop"]) != normalize(args.early_stop):
            return False

    if normalize(args.client_language_composition) != "all":
        if metadata["composition"] is None:
            return False
        if normalize(metadata["composition"]) != normalize(args.client_language_composition):
            return False

    return True


def process_log_file(log_path, out_dir, metadata, args):
    last_round = extract_last_round_fl_training(log_path)
    losses = extract_losses_from_log(log_path)
    if not losses["clients"]:
        print(f"[WARN] No client losses found in {log_path.name}. Skipping.")
        return

    num_rounds = infer_num_rounds(losses, last_round)
    if num_rounds <= 0:
        print(f"[WARN] Could not infer rounds for {log_path.name}. Skipping.")
        return

    clients = metadata["clients"] if metadata["clients"] is not None else args.c
    patience = metadata["patience"] if metadata["patience"] is not None else args.patience
    early_stop = metadata["early_stop"] if metadata["early_stop"] is not None else args.early_stop
    fl_method = metadata["fl_method"] if metadata["fl_method"] is not None else args.fl_method
    composition = metadata["composition"] if metadata["composition"] is not None else "unknown_composition"

    losses = add_none_when_local_early_stop(losses, num_rounds)
    client_order, labels = get_row_order_and_labels(losses["clients"].keys(), composition)

    output_suffix = f"{clients}c_pat{patience}_{early_stop}_{fl_method}_{composition}"

    train_plot_path = out_dir / f"clients_evolution_diagram_{output_suffix}.png"
    matrix_plot_path = out_dir / f"influence_matrix_{output_suffix}.png"

    plot_train_diagram(losses, client_order, labels, num_rounds, train_plot_path)
    influence_matrix = build_influence_matrix(losses, client_order)
    plot_influence_matrix(influence_matrix, labels, matrix_plot_path)

    print(
        f"[OK] {log_path.name} -> {train_plot_path.name}, {matrix_plot_path.name}"
    )


def main():
    parser = argparse.ArgumentParser(description="Plot training diagram and influence matrix")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--early_stop", type=str, required=True)
    parser.add_argument("--fl_method", type=str, required=True)
    parser.add_argument("--client_language_composition", type=str, required=True)

    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    log_dir = work_dir / "exp_logs" / args.dataset / f"{args.c}c" / args.model_name
    out_dir = work_dir / "plots" / args.dataset / f"{args.c}c" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No .log files found in: {log_dir}")

    matched_logs = []
    for log_file in log_files:
        metadata = parse_log_metadata(log_file)
        if not log_matches_filters(metadata, args):
            continue
        matched_logs.append((log_file, metadata))

    if not matched_logs:
        raise FileNotFoundError(
            "No matching logs found with the requested filters in "
            f"{log_dir} (c={args.c}, patience={args.patience}, "
            f"early_stop={args.early_stop}, fl_method={args.fl_method}, "
            f"client_language_composition={args.client_language_composition})"
        )

    for log_file, metadata in matched_logs:
        process_log_file(log_file, out_dir, metadata, args)


if __name__ == "__main__":
    main()
