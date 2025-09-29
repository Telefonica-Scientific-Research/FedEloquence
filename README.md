# FedEloquence - ICASSP Branch

## Overview

This branch contains the experimental code and configurations for our paper: **"CLIENT-DRIVEN CONVERGENCE IN FEDERATED LEARNING OF MULTILINGUAL LARGE LANGUAGE MODELS"**.

Our contributions extend the FederatedScope repository to support **multilingual federated fine-tuning of Large Language Models (LLMs)**.

> **Note:** For general installation and setup instructions, please refer to the **main branch** of the repository.

---

## Key Features

### 🌍 Multilingual Fine-tuning Support

- **Flexible prompt integration** – Easily add new prompts for different languages
- **Language-aware processing** – Sample-wise processing based on language tags
- **Comprehensive data pipeline** – Tools for creating multilingual FL partitions including:
  - Server-side validation and test sets
  - Client-side train, validation and test sets
- **Evaluation metrics** – Built-in scripts for computing **BERTScore** and **ROUGE**

### 🎯 Local Dynamic Early Stop for Federated Learning (LDES-FL)

Our novel early stopping mechanism that allows clients to dynamically pause and resume training.

**Configuration:**

Set `federate.use_LDES` to `true` and `federate.use_global_early_stop` to `false` in your config file.

### 📊 FedValLoss Aggregation Method

A validation loss-based aggregation strategy for improved convergence.

**Configuration:**

Set `federate.method` to `FedValLoss` in your configuration.

### ✨ Additional Improvements

- Flexible evaluation options against server and/or client test/validation sets
- Enhanced logging and monitoring capabilities
- Optimized data loading for multilingual scenarios

---

## Running Experiments

### Prerequisites

Ensure you have:
- Activated your virtual environment
- Configured the `ds_config` file with appropriate training parameters for your hardware

### Configuration Files

All experimental configurations are located in `configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/`:

#### Early Stop Experiments
- `early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedavg.yaml`
- `early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedprox.yaml`
- `early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedvalloss.yaml`

#### Global (Language-Specific) Experiments
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_CA.yaml` (Catalan)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_DA.yaml` (Danish)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_DE.yaml` (German)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_EN.yaml` (English)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_ES.yaml` (Spanish)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_EU.yaml` (Basque)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_HR.yaml` (Croatian)
- `global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_SR.yaml` (Serbian)

### Execution

Run the following command, replacing the config path with your desired experiment configuration:

    deepspeed federatedscope/main.py --cfg configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedavg.yaml

---

## Dataset Creation

### Multilingual FL Dataset Pipeline

To create the multilingual dataset for federated learning:

1. **Navigate to the dataset creation script:**
   
   `create_FL_multilingual_datasets/create_dataset_alpaca_cleaned_8c`

2. **Configure the script** according to your requirements

3. **Save the `.jsonl` files** for all languages participating in the federation in:

   `create_FL_multilingual_datasets/{dataset}/jsonl`

   Use the naming convention `{lang_tag}.jsonl` — e.g., `ca.jsonl`, `es.jsonl`, `en.jsonl`, etc.

4. **Run the script** to generate a `.jsonl` file containing the multilingual data

5. **Automatic partitioning:** FedEloquence will automatically pick the data from the partition done in this step:
   - Server validation and test sets
   - Client training, validation, and test sets

---

## Visualization Tools

### 📈 Client Training Analysis

Visualize client participation and interaction patterns during LDES-FL training from resulting logs.

#### Available Visualizations

1. **Client Evolution Diagram**
   - Timeline of client activity across training rounds
   - Visual indicators for local early stopping and training resumption events

2. **Client-to-Client Influence Matrix**
   - Heatmap showing inter-client influence
   - Tracks how often each client contributes to another's training resumption

#### Setup & Execution

1. **Prepare your logs:**
   
   Place logs in: `analysis/exp_logs/{model}/{dataset}_{n_clients}c/`
   
   Name each log file according to the aggregation method used (e.g., `fedavg.log`, `fedprox.log`)

2. **Configure parameters** in the script (if needed):

  `analysis/plot_diagrams.sh`

3. **Generate visualizations:**
   
   Run: `sh analysis/plot_diagrams.sh`

4. **View results:**
   
   Output will be saved in: `analysis/plots/{model}/{dataset}_{n_clients}c/`

---

## Configuration Naming Convention

The configuration filenames follow this pattern:

    ds_{clients}_{rounds}_{local_steps}_r{rank}_a{alpha}_{patience}_{training_steps}_{method}.yaml

**Example:** `ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedavg.yaml`
- `8c` = 8 clients
- `5000r` = 5000 max communication rounds
- `30ls` = 30 local steps
- `r16` = LoRA rank 16
- `a32` = LoRA alpha 32
- `1pat` = patience 1
- `160lts` = 160 local training steps
- `fedavg` = FedAvg aggregation method

---

## Support

For issues related to:
- **Installation/Setup:** See the main branch documentation
- **Experiments:** Open an issue in this repository
- **FederatedScope:** Refer to the [FederatedScope repository](https://github.com/alibaba/FederatedScope)

---

## Citation

If you use this code in your research, please cite our paper:

    @inproceedings{feedeloquence2025,
      title={Client-Driven Convergence in Federated Learning of Multilingual Large Language Models},
      author={Your Name et al.},
      booktitle={ICASSP 2025},
      year={2025}
    }

---

## License

[Specify your license here]