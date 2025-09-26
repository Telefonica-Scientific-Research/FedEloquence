## Installation, Setup, and Running of **FedEloquence** for LLM Fine-tuning

Please refer to the **main branch** of the repository for installation and setup instructions of the framework.

---

## Content of this Branch (`icassp`)

This branch contains the scripts and instructions required to reproduce the results presented in our paper *CLIENT-DRIVEN CONVERGENCE IN FEDERATED LEARNING OF MULTILINGUAL LARGE LANGUAGE MODELS*.  

Our contributions extend the FederatedScope repository to support **multilingual federated fine-tuning of LLMs**.  

Specifically, this branch provides:  

- **Multilingual fine-tuning support**  
  - Flexible integration of new prompts.  
  - Sample-wise processing based on language tags.  
  - Data preprocessing pipelines for creating multilingual FL partitions (train, validation and test sets for both server and clients).  
  - Evaluation scripts computing **BERTScore** and **ROUGE**.  

- **Local Dynamic Early Stop for Federated Learning (LDES-FL)**  
  - To enable LDES-FL, set federate.use_local_early_stop to `True` in the config file (and federate.use_global_early_stop to False to disable).  

- **FedValLoss aggregation method**  
  - To activate this method, set `federate.method = FedValLoss` in the configuration.  

- **Additional improvements**  
  - Option to evaluate against the server and/or clients’ test/val sets.  

- **Configuration files** (`.yaml`) used in our experiments:
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedavg.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedprox.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedvalloss.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_CA.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_DA.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_DE.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_EN.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_ES.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_EU.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_HR.yaml
    - configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/global/ds_8c_5000r_r16_a32_3pat_1eval_3b_16320lts_SR.yaml
    
    Once in the virtual environment, to run these scripts do: 
    deepspeed federatedscope/main.py --cfg configs/multilingual/alpaca_cleaned/8c/salamandra-2b-instruct/early_stop/ds_8c_5000r_30ls_r16_a32_1pat_160lts_fedavg.yaml
    It is important to set the ds_config file to the correct training parameters of your machines.

### Client Training Diagram & Influence Matrix Plotter

In this section we present the utilities to **visualize client participation and interactions** during training with **Local Dynamic Early Stop in Federated Learning (LDES-FL)**.

We can generate **two key visualizations**:

1. **Client Evolution Diagram**  
   A timeline showing each client's activity across training rounds, including when they perform **local early stop** and **resume training**.

2. **Client-to-Client Influence Matrix**  
   A heatmap illustrating how often each client contributes to another client's resumption of training after local early stopping.
   
---

## 🔧 How It Works

### 1. Prepare Experiment Logs

Place your experiment logs in the following directory structure: "analysis/exp_logs/{model}/{dataset}_{n_clients}c/".
Name each log file according to the **aggregation method** used.

### 2. Configure Parameters & Run the Script

Launch the visualization process using the provided bash script:

```bash
sh analysis/plot_diagrams.sh
```

### 3. View the Generated Plots
The output plots will be saved in: "analysis/plots/{model}/{dataset}_{n_clients}c/"