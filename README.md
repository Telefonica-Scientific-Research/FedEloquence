# Installation, Setup and Running of FederatedScope for LLMs Fine-tuning

First, use a virtual environment manager such as uv to create a virtual environment. Make sure you are using Python 3.9.0:
To install uv please follow the next guidelines: https://docs.astral.sh/uv/getting-started/installation or just run the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv FedEloq --python 3.9.0
source FedEloq/bin/activate
```

Clone the specific branch of the FederatedScope repository to your machine:

```bash
git clone https://github.com/aleixsant/FedEloquence.git
```

Open the FedEloquence directory and build the package to obtain a distribution file:
```bash
cd FedEloquence
uv build
```

With the package built, you can now install it locally to test it out. Use the following command:
```bash
uv pip install dist/federatedscope-0.3.0-py3-none-any.whl
```

If you're developing the package and want to make sure changes are immediately reflected without rebuilding every time, you can also install in editable mode: 

```bash
uv pip install -e .
```

Before using DeepSpeed, review the configuration file at `federatedscope/llm/deepspeed/ds_config.json`. Ensure that the train_batch_size parameter is properly set to match the number of GPUs available on your machine (default: "train_batch_size": 1).

Check if fine-tuning an LLM in standalone mode works correctly with DeepSpeed. Run the following script to verify that the fine-tuning process is functioning properly:

```bash
deepspeed federatedscope/main.py --cfg configs/standalone/occiglot-7B-eu5-instruct/ds_3c_200r_30ls.yaml
```

To execute federated fine-tuning in distributed mode, separate commands need to be run for the server and each client. In the FederatedScope framework, each client must run on a different machine. The following config files will allow us to test if the setup works with two clients in distributed mode. However, before running the commands, ensure that the `server_host`, `server_port`, `client_host`, and `client_port` fields in the config files are updated with the correct IP addresses and ports for your machines. Additionally, adjust CUDA_VISIBLE_DEVICES to reflect the number of GPUs available on each machine.

To run the server use:
```bash
deepspeed --master_addr=127.0.0.1 --master_port=29500 federatedscope/main.py --cfg configs/distributed/Phi-3.5-mini-instruct/server_ds_2c_200r_30ls.yaml
```

To run a first client in one machine use:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --master_addr=127.0.0.1 --master_port=29500 federatedscope/main.py --cfg configs/distributed/Phi-3.5-mini-instruct/client_1_ds_2c_200r_30ls.yaml
```

To run a second client in another machine:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --master_addr=127.0.0.1 --master_port=29500 federatedscope/main.py --cfg configs/distributed/Phi-3.5-mini-instruct/client_2_ds_2c_200r_30ls.yaml 
```

To ensure that the correct CUDA paths are set, add the following lines to your `.bashrc` (or equivalent shell configuration file). The CUDA version should be around version 12 (e.g., 12.4, 12.5, or 12.6). If you don’t already have the [CUTLASS](https://github.com/NVIDIA/cutlass) repository installed, clone and set it up on your machine.

```bash
export PATH=/usr/local/cuda-12/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:/usr/local/cuda-12/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12
export CUTLASS_PATH=/home/user/repos/cutlass 
```

After editing `.bashrc`, don't forget to run:

```bash
source ~/.bashrc
```