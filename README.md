# Installation, Setup, and Running of **FedEloquence** for LLM Fine-tuning

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