import os
import gzip
import json
import random
import logging
import torch
import transformers
import importlib

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, LLMDataset
from federatedscope.core.data.utils import download_url

logger = logging.getLogger(__name__)


@dataclass
class LLMDataCollator(object):
    """
    A data collator for supervised fine-tuning of language models.
    This class implements a callable that takes a list of instances and
    returns a batch of input_ids, labels, and attention_mask tensors. The
    input_ids and labels are padded with the tokenizer's pad_token_id and a
    special ignore index value, respectively. The attention_mask indicates
    which tokens are not padding.
    """

    tokenizer: transformers.PreTrainedTokenizer
    debug: bool = False
    debug_max_samples: int = 3 # Maximum is the batch size

    def __call__(self, instances):
        """Collate a list of variable-length samples into a padded batch
        for supervised fine-tuning of LLMs.

        If in tokenization, padding="max_length" was used, the sequences are
        already padded to the max length, so no further padding (dynamic) is needed here.
        """

        # ---- Extract fields ----
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]

        # ---- Optional: log original lengths ----
        if self.debug:
            orig_lengths = [len(x) for x in input_ids_list]
            logger.info(f"Original lengths in batch: {orig_lengths}")

        # Pad sequences to the max length in the batch (dynamic padding)
        # Trucation with max_length is handled previously in the Dataset class (llm_dataset.py)
        # ---- Dynamic padding ----
        pad_id = self.tokenizer.pad_token_id
        ignore_id = DefaultToken.IGNORE_INDEX.value
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=ignore_id,
        )

        # ---- Create attention mask ---
        # Attention masks depend on batch padding
        # attention_mask = 0 → token is invisible to attention
        # attention_mask = 1 → token is attended to
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        # ---- Optional sanity checks ----
        if self.debug:
            logger.info(f"Padded batch shape: {input_ids.shape}")

            pad_positions = input_ids == pad_id

            assert torch.all(attention_mask[pad_positions] == 0), \
                "Padding tokens must have attention_mask = 0"

            assert torch.all(labels[pad_positions] == ignore_id), \
                "Padding labels must be IGNORE_INDEX"

            assert torch.all(attention_mask[~pad_positions] == 1), \
                "Non-padding tokens must have attention_mask = 1"

            assert input_ids.shape == labels.shape == attention_mask.shape
            assert input_ids.dtype == torch.long
            assert labels.dtype == torch.long
            assert attention_mask.dtype == torch.long

            # ---- Print a few samples (truncated) ----
            bs = input_ids.size(0) # batch size
            for i in range(min(self.debug_max_samples, bs)):
                logger.info(f"\n=== Batch sample {i} ===")
                logger.info(f"input_ids: {input_ids[i].tolist()}")
                logger.info(f"labels: {labels[i].tolist()}")
                logger.info(f"attention_mask: {attention_mask[i].tolist()}")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

def get_tokenizer(model_name, cache_dir, tok_len=128, pkg='huggingface_llm'):
    """
    This function loads a tokenizer from a pretrained model name and adds some
    default special tokens if they are not already defined. It also sets the
    model max length and the padding side of the tokenizer.

    Args:
        model_name: A string, the name of the pretrained model.
        cache_dir: A string, the path to the cache directory.
        tok_len: An integer, the maximum length of the tokens. Defaults to 128.

    Returns:
        A tuple of (tokenizer, num_new_tokens), where:
            - tokenizer: A transformers.AutoTokenizer object.
            - num_new_tokens: An integer, the number of new special tokens
    """
    assert pkg in ['huggingface_llm', 'modelscope_llm'], \
        f'Not supported package {pkg}.'

    if pkg == 'huggingface_llm':
        from transformers import AutoTokenizer
    elif pkg == 'modelscope_llm':
        from modelscope import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=tok_len,
        padding_side="right",
        use_fast=True,
    )

    special_tokens = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.unk_token:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category',
              language='language'):
    """
    This function reads a JSON file that contains a list of examples,
    each with an instruction, an input, an output, and a category. It
    returns a list of dictionaries with the same keys, but with the
    option to rename them.

    Args:
        file_path: A string, the path to the JSON file.
        instruction: A string, the key for the instruction field. Defaults
            to 'instruction'.
        input: A string, the key for the input field. Defaults to 'input'.
        output: A string, the key for the output field. Defaults to 'output'.
        category: A string, the key for the category field. Defaults to
            'category'.

    Returns:
        A list of dictionaries, each with four keys: instruction, input,
            output, and category. The values are taken from the JSON file
            and may be None if the corresponding key is not present in the
            file.
    """

    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None,
            language=item[language] if language in item else None)
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               language='language',
               is_gzip=False):
    """
    This function reads a JSONL file that contains one example per line,
    each with an instruction, an input, an output, and a category. It
    returns a list of dictionaries with the same keys, but with the option
    to rename them. It also supports reading gzip-compressed files.

    Args:
        file_path: A string, the path to the JSONL file.
        instruction: A string, the key for the instruction field. Defaults
            to 'instruction'.
        input: A string, the key for the input field. Defaults to 'input'.
        output: A string, the key for the output field. Defaults to 'output'.
        category: A string, the key for the category field. Defaults to
            'category'.
        is_gzip: A boolean, whether the file is gzip-compressed or not.
            Defaults to False.

    Returns:
        A list of dictionaries, each with four keys: instruction, input,
        output, and category. The values are taken from the JSONL file and
        may be None if the corresponding key is not present in the line.

    """
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
                language=item[language] if language in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_llm_dataset(config=None, **kwargs):
    """
    This function takes a config object and optional keyword arguments and
    returns a dataset object and an updated config object.
    The function supports various dataset types, such as JSON, JSONL, alpaca,
    alpaca_cleaned, dolly-15K, gsm8k, code_search_net, rosetta_alpaca. It
    will download the data files from their respective URLs if they are not
    found in the data directory. It will also load a tokenizer from a
    pretrained model name and add some default special tokens if they are
    not already defined.

    Args:
        config: An object, the configuration for loading the dataset.
        **kwargs: Optional keyword arguments that can override the config
            attributes.

    Returns:
        A tuple of (dataset, config), where:
            - dataset: A LLMDataset object that contains the examples with
                instruction, input, output, and category fields.
            - config: An object, the updated configuration.
    """
    # For debugging purposes
    inspect_raw_prompt = False
    inspect_masking = False

    model_name, model_hub = config.model.type.split('@')
    
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len,
                      model_hub)

    dataset_name, _ = config.data.type.split('@')

    domain_type = config.data.domain_type  

    path_to_prompt = config.llm.prompt_path
    module_path = path_to_prompt.replace("/", ".")

    module = importlib.import_module(module_path)
    PROMPT = module.PROMPT

    if dataset_name.endswith('.json'):
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
    elif dataset_name.endswith('.jsonl'):
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_jsonl(fp)
        if inspect_raw_prompt:
            inspect_raw_prompt(list_data_dict[0], PROMPT)
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
        if inspect_masking:
            inspect_masking(dataset, tokenizer, index=0)
    elif dataset_name.lower() == 'alpaca':
        fp = os.path.join(config.data.root, 'alpaca_data.json')
        download_url(
            'https://raw.githubusercontent.com/tatsu-lab'
            '/stanford_alpaca/'
            '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
            'alpaca_data.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
    elif dataset_name.lower() == 'alpaca_cleaned':
        fp = os.path.join(config.data.root, 'alpaca_data_cleaned.json')
        download_url(
            'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/'
            'a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/'
            'alpaca_data_cleaned.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
    elif dataset_name.lower() == 'dolly-15k':
        fp = os.path.join(config.data.root, 'databricks-dolly-15k.jsonl')
        download_url(
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl', config.data.root)
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='context',
                                    output='response',
                                    category='category')
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
    elif dataset_name.lower() == 'gsm8k':
        fp = os.path.join(config.data.root, 'gsm8k_train.jsonl')
        if not os.path.exists(fp):
            download_url(
                'https://raw.githubusercontent.com/openai/grade-school-math'
                '/3101c7d5072418e28b9008a6636bde82a006892c/'
                'grade_school_math/data/train.jsonl', config.data.root)
            os.rename(os.path.join(config.data.root, 'train.jsonl'), fp)
        list_data_dict = load_jsonl(fp,
                                    instruction='question',
                                    output='answer')
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('####', 'The answer is')
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
    elif dataset_name.lower() == 'code_search_net':
        from tqdm import tqdm
        from federatedscope.llm.dataset.code_search_net import \
            CSN_FILE_NUM_DICT

        list_data_dict = []
        logger.info('Loading code search net data file...')
        try:
            for language in tqdm(CSN_FILE_NUM_DICT.keys()):
                sub_list_data_dict = []
                for file_index in range(CSN_FILE_NUM_DICT[language]['train']):
                    fp = \
                        os.path.join(config.data.root, language,
                                     'final', 'jsonl', 'train',
                                     f'{language}_train_{file_index}.jsonl.gz')
                    tmp_list_data_dict = load_jsonl(
                        fp,
                        instruction='docstring',
                        input='language',
                        output='code',
                        category='language',
                        is_gzip=True,
                    )
                    sub_list_data_dict += tmp_list_data_dict
                # Subsample
                raw_size = len(sub_list_data_dict)
                num_subsample = int(raw_size * config.data.subsample)
                list_data_dict += random.sample(sub_list_data_dict,
                                                num_subsample)
                logger.info(f"Subsample "
                            f"{sub_list_data_dict[0]['category']} with "
                            f"rate {config.data.subsample}: "
                            f"the sample size is # {num_subsample} "
                            f"(the raw size is {raw_size}).")
            # Modify instruction with specific language
            for sample in list_data_dict:
                sample['instruction'] = \
                    sample['category'] + ' ' + sample['instruction']
        except FileNotFoundError:
            raise FileNotFoundError(
                'Data not found! Please run `python '
                'federatedscope/llm/dataset/code_search_net.py` '
                'to download data.')
        dataset = LLMDataset(list_data_dict, tokenizer, PROMPT)
    elif dataset_name.lower() == 'rosetta_alpaca':
        fp = os.path.join(config.data.root, 'rosetta_alpaca.json')
        download_url(
            'https://raw.githubusercontent.com/'
            'sahil280114/codealpaca/'
            'd269da106a579a623a654529b3cb91b5dfa9c72f/'
            'data/rosetta_alpaca.json', config.data.root)
        list_data_dict = load_json(fp,
                                   instruction='instruction',
                                   input='input',
                                   output='output',
                                   category='input')
        # Remove 'x86-64 Assembl' if splitter is `meta` due to the number of
        # samples is too small.
        if config.data.splitter == 'meta':
            list_data_dict = [
                i for i in list_data_dict if i['category'] != 'X86-64 Assembly'
            ]
        dataset = LLMDataset(list_data_dict, tokenizer, domain_type, PROMPT)
    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    return dataset, config


def load_llm_dir_dataset(config=None, client_cfgs=None, **kwargs):
    """
    Load a pre-partitioned FL LLM dataset from a directory structure.

    Expected layout (example):
      <base_dir>/server/val.jsonl            (optional)
      <base_dir>/server/test.jsonl           (optional)
      <base_dir>/1c/train.jsonl              (required)
      <base_dir>/1c/val.jsonl                (required)
      <base_dir>/1c/test.jsonl               (optional)
      ...
      <base_dir>/<Nc>/train.jsonl            (required), where N=client_num

    Notes:
      - Server train split is ignored if provided.
      - Client folders can be named as "<id>c" or "c<id>".
      - This function builds and returns ``StandaloneDataDict`` directly.
    """
    from federatedscope.core.data import ClientData, StandaloneDataDict

    if config is None:
        raise ValueError("`config` is required for `load_llm_dir_dataset`.")

    model_name, model_hub = config.model.type.split('@')
    tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                 config.llm.tok_len, model_hub)

    path_to_prompt = config.llm.prompt_path
    module_path = path_to_prompt.replace("/", ".")
    module = importlib.import_module(module_path)
    prompt = module.PROMPT
    domain_type = config.data.domain_type

    base_dir = config.data.file_path
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(config.data.root, base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Pre-partitioned data dir not found: {base_dir}")
    logger.info(f"[dir_jsonl@llm] Loading pre-partitioned data from: {base_dir}")

    def _resolve_client_dir(base_path, cid):
        candidates = [
            os.path.join(base_path, f"{cid}c"),
            os.path.join(base_path, f"c{cid}"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                return cand
        return None

    def _parse_jsonl_with_validation(split_path):
        list_data_dict = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if line == "":
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {split_path} at line {line_no}: {e.msg}"
                    ) from e
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Each JSONL row must be an object in {split_path}, "
                        f"line {line_no}.")

                if item.get('instruction') in [None, ""]:
                    raise ValueError(
                        f"Missing required key `instruction` in {split_path}, "
                        f"line {line_no}.")
                if item.get('output') in [None, ""]:
                    raise ValueError(
                        f"Missing required key `output` in {split_path}, "
                        f"line {line_no}.")
                if domain_type == "cross" and item.get('language') in [None, ""]:
                    raise ValueError(
                        f"`language` is required when data.domain_type='cross' "
                        f"in {split_path}, line {line_no}.")

                new_item = dict(
                    instruction=item['instruction'],
                    input=item['input'] if 'input' in item else None,
                    output=item['output'],
                    category=item['category'] if 'category' in item else None,
                    language=item['language'] if 'language' in item else None,
                )
                list_data_dict.append(new_item)
        return list_data_dict

    def _load_split(split_path, required=False):
        if not os.path.isfile(split_path):
            if required:
                raise FileNotFoundError(f"Required split file not found: {split_path}")
            return None

        list_data_dict = _parse_jsonl_with_validation(split_path)
        if len(list_data_dict) == 0:
            if required:
                raise ValueError(f"Required split file is empty: {split_path}")
            logger.warning(f"Optional split file is empty: {split_path}")
        return LLMDataset(list_data_dict, tokenizer, domain_type, prompt)

    # Server data: val/test are optional
    server_dir = os.path.join(base_dir, "server")
    server_val = None
    server_test = None
    if os.path.isdir(server_dir):
        server_val = _load_split(os.path.join(server_dir, "val.jsonl"),
                                 required=False)
        server_test = _load_split(os.path.join(server_dir, "test.jsonl"),
                                  required=False)
        if os.path.isfile(os.path.join(server_dir, "train.jsonl")):
            logger.warning(
                "Found server/train.jsonl, but nothing is implemented for "
                "server train split in this pipeline. The file is currently "
                "not used.")
    else:
        logger.warning(f"No `server` folder found under {base_dir}; "
                       f"server will have no val/test data.")

    data_dict = {0: ClientData(config, train=None, val=server_val, test=server_test)}
    summary_rows = []
    summary_rows.append({
        "participant": "server",
        "train": 0,
        "val": len(server_val) if server_val is not None else 0,
        "test": len(server_test) if server_test is not None else 0,
        "path": server_dir if os.path.isdir(server_dir) else "-"
    })

    # Client data: train/val required, test optional
    client_num = config.federate.client_num
    recognized_client_dirs = set()
    for client_id in range(1, client_num + 1):
        client_dir = _resolve_client_dir(base_dir, client_id)
        if client_dir is None:
            raise FileNotFoundError(
                f"Client folder not found for client {client_id}. "
                f"Expected one of: {client_id}c/ or c{client_id}/ "
                f"under {base_dir}")
        recognized_client_dirs.add(os.path.abspath(client_dir))

        train_set = _load_split(os.path.join(client_dir, "train.jsonl"),
                                required=True)
        val_set = _load_split(os.path.join(client_dir, "val.jsonl"),
                              required=True)
        test_set = _load_split(os.path.join(client_dir, "test.jsonl"),
                               required=False)

        if client_cfgs is not None:
            client_cfg = config.clone()
            client_cfg.merge_from_other_cfg(client_cfgs.get(
                f'client_{client_id}'))
        else:
            client_cfg = config

        data_dict[client_id] = ClientData(client_cfg,
                                          train=train_set,
                                          val=val_set,
                                          test=test_set)

        train_n = len(train_set) if train_set is not None else 0
        val_n = len(val_set) if val_set is not None else 0
        test_n = len(test_set) if test_set is not None else 0
        logger.info(f"Loaded client {client_id} from {client_dir}: "
                    f"train={train_n}, val={val_n}, test={test_n}")
        summary_rows.append({
            "participant": f"client_{client_id}",
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "path": client_dir
        })

    # Warn if there are client-like folders in dataset dir that are unused.
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.lower() == "server":
            continue
        is_client_like = entry.endswith('c') and entry[:-1].isdigit()
        is_client_like = is_client_like or (entry.startswith('c') and entry[1:].isdigit())
        if is_client_like and os.path.abspath(entry_path) not in recognized_client_dirs:
            logger.warning(
                f"Found unreferenced client folder `{entry}` under {base_dir}. "
                f"It is ignored because federate.client_num={client_num}.")

    server_val_n = len(server_val) if server_val is not None else 0
    server_test_n = len(server_test) if server_test is not None else 0
    logger.info(f"Loaded server from {server_dir}: "
                f"val={server_val_n}, test={server_test_n}")

    total_train = sum(row["train"] for row in summary_rows)
    total_val = sum(row["val"] for row in summary_rows)
    total_test = sum(row["test"] for row in summary_rows)

    # Compact summary table for quick inspection.
    header = f"{'participant':<12} {'train':>10} {'val':>10} {'test':>10}  path"
    logger.info("[dir_jsonl@llm] Dataset summary:")
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary_rows:
        logger.info(
            f"{row['participant']:<12} {row['train']:>10} {row['val']:>10} "
            f"{row['test']:>10}  {row['path']}")
    logger.info("-" * len(header))
    logger.info(
        f"{'TOTAL':<12} {total_train:>10} {total_val:>10} {total_test:>10}")

    return StandaloneDataDict(data_dict, config), config

def inspect_raw_prompt(example, prompt_dict):
    """
    Visualize how an example is converted into a formatted prompt
    BEFORE ChatML wrapping.

    Supports:
      - multilingual prompt templates
      - input/no-input branches
    """

    language = example.get("language")
    if language is None:
        raise ValueError("Example must contain a 'language' key.") # language is the lang tag

    # Pick the correct language prompt template (de, en, ca, etc.)
    lang_prompts = prompt_dict.get(language)
    if lang_prompts is None:
        raise ValueError(f"No prompt template found for language '{language}'")

    prompt_input = lang_prompts.get("prompt_input")
    prompt_no_input = lang_prompts.get("prompt_no_input")

    # Select input vs no-input template
    if example.get("input") not in ("", None):
        source = prompt_input.format_map(example)
    else:
        source = prompt_no_input.format_map(example)

    # Target normally does NOT include EOS here
    target = example["output"]

    logger.info("\n==============================")
    logger.info(" RAW SOURCE (before ChatML) ")
    logger.info("==============================")
    logger.info(source)

    logger.info("\n==============================")
    logger.info(" RAW TARGET (before ChatML) ")
    logger.info("==============================")
    logger.info(target)

    # If padding="max_length" was used during tokenization,
    # we would see MASK tokens in the system+user tokens and
    # the padding tokens.
    logger.info("\n==============================")
    logger.info(" CONCATENATED (source + target) ")
    logger.info("==============================")
    logger.info(source)
    logger.info(target)

    return source, target

def inspect_masking(dataset, tokenizer, index=0, show_tokens=False, skip_padding=True):
    """
    Visualize a sample's input_ids vs labels with masking applied.

    Args:
        dataset: The dataset object (LLMDataset).
        tokenizer: The HF tokenizer used to encode the data.
        index: Index of the sample to inspect.
        show_tokens: If True, prints token-level comparison of input vs label masking.
        skip_padding: If True, skips pad tokens when showing masked labels.
    """
    item = dataset[index]

    input_ids = item["input_ids"]
    labels = item["labels"]

    # Convert tensors to lists for easier processing
    input_ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else labels

    # Replace masked tokens (-100) with pad token for decoding
    masked_labels = [
        tok_id if tok_id != -100 else tokenizer.pad_token_id
        for tok_id in labels_list
    ]

    if skip_padding:
        masked_labels_filtered = [tok_id for tok_id in masked_labels if tok_id != tokenizer.pad_token_id]
    else:
        masked_labels_filtered = masked_labels

    logger.info("\n=== DECODED INPUT IDS (full example) ===")
    logger.info(tokenizer.decode(input_ids_list, skip_special_tokens=False))

    logger.info("\n=== DECODED LABELS (only unmasked parts) ===")
    logger.info(tokenizer.decode(masked_labels_filtered, skip_special_tokens=False))

    logger.info("\n=== MASK POSITIONS ===")
    mask_pos = ["MASK" if lbl == -100 else lbl for lbl in labels_list]
    logger.info(mask_pos)

    if show_tokens:
        logger.info("\n=== TOKEN-LEVEL COMPARISON ===")
        for inp_tok, lbl_tok in zip(input_ids_list, labels_list):
            inp_str = tokenizer.decode([inp_tok])
            mark = "MASK" if lbl_tok == -100 else "PRED"
            logger.info(f"{inp_str:>10}: {mark}")
