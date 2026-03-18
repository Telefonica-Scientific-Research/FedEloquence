"""
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

import copy
import logging
import pandas as pd

from enum import Enum
from torch.utils.data import Dataset
from datetime import datetime

logger = logging.getLogger(__name__)


class DefaultToken(Enum):
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}

class LLMDataset(Dataset):
    """
    A dataset for language modeling tasks.

    This class inherits from torch.utils.data.Dataset and implements a
    dataset that can load and preprocess data for language modeling. It
    takes a list of data dictionaries, a tokenizer, and optional prompt
    templates as input, and creates input ids, labels, and categories as
    output. The input ids and labels are padded and masked according to
    the tokenizer settings and the source and target lengths. The
    categories are encoded as integers using pandas.Categorical.

    Attributes:
        input_ids: A list of torch.LongTensor objects of shape (max_length,)
            containing the padded input ids.
        labels: A list of torch.LongTensor objects of shape (max_length,)
            containing the padded labels.
        categories: A list of integers representing the category codes.
        tokenizer: A transformers.PreTrainedTokenizer object that can
            encode and decode text.
    """
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 domain_type,
                 prompt):
        """
        Initializes the dataset with the given arguments.

        Args:
            list_data_dict: A list of dictionaries, each containing input,
                output, and optionally category keys and values as strings.
            tokenizer: A transformers.PreTrainedTokenizer object that can
                encode and decode text.
            prompt_input: An optional string template for creating the source
                text when the input key is present in the data dictionary.
                The template can use {input}, {output}, and {category} as
                placeholders for the corresponding values. The default value
                is PROMPT_DICT["prompt_input"].
            prompt_no_input: An optional string template for creating the
                source text when the input key is not present in the data
                dictionary. The template can use {output} and {category} as
                placeholders for the corresponding values. The default value is
                PROMPT_DICT["prompt_no_input"].
        """
        super(LLMDataset, self).__init__()

        # ------------------- Ensure proper pad_token -------------------
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.unk_token:
            tokenizer.add_special_tokens({"pad_token": DefaultToken.PAD_TOKEN.value})

        self.tokenizer = tokenizer

        sources, targets = [], []

        if domain_type == "single":
            prompt_input, prompt_no_input = prompt.get("prompt_input"), prompt.get("prompt_no_input")
            for example in list_data_dict:
                formatted_source = (
                    prompt_input.format_map(example) 
                    if example.get("input") not in ("", None) 
                    else prompt_no_input.format_map(example)
                )
                sources.append(formatted_source)
                targets.append(f"{example['output']}{tokenizer.eos_token}")
        else:  # cross
            for example in list_data_dict:
                lang_prompts = prompt.get(example.get("language"), {})
                formatted_source = (
                    lang_prompts.get("prompt_input", "").format_map(example) 
                    if example.get("input") not in ("", None) 
                    else lang_prompts.get("prompt_no_input", "").format_map(example)
                )
                sources.append(formatted_source)
                targets.append(f"{example['output']}{tokenizer.eos_token}")

        data_dict = self.preprocess(sources, targets)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

    def _tokenize_fn(self, list_of_messages):
        """
        Tokenizes a list of chat message sequences using the tokenizer's
        chat template correctly for instruction tuning.

        Args:
            list_of_messages: list of conversations, where each element is:
                [
                    {"role": "user", "content": source_text},
                    {"role": "assistant", "content": target_text},
                ]
            tokenizer: HF PreTrainedTokenizer with chat template.

        Returns:
            dict containing:
                input_ids: list of LongTensor
                labels:    list of LongTensor
                input_ids_lens: list[int]
                labels_lens: list[int]
        """

        input_ids_list = []
        labels_list = []
        input_lens = []
        label_lens = []

        tokenizer = self.tokenizer

        pad_id = tokenizer.pad_token_id
        max_len = tokenizer.model_max_length

        date_str = datetime.today().strftime("%Y-%m-%d")

        for messages in list_of_messages:
            # Render full conversation using chat template
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                date_string=date_str
            )

            # Tokenize the full sequence
            encoded = tokenizer(
                rendered,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
                return_tensors="pt"
            )
            # We don't apply padding here. We will do dynamic padding in the DataCollator
            #  padding="max_length", if it was like this we would increase the sample to max_lens with zeros (padding)

            input_ids = encoded.input_ids[0]
            labels = input_ids.clone()

            # ---------------- Mask user/system tokens (leaving assistant tokens only) ----------------
            # Identify all messages BEFORE the assistant
            prefix_msgs = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    break
                prefix_msgs.append(msg)

            if prefix_msgs:
                rendered_prefix = tokenizer.apply_chat_template(
                    prefix_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    date_string=date_str,
                )
                prefix_ids = tokenizer(
                    rendered_prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).input_ids[0]
                # Mask the prefix tokens in labels (system + user)
                labels[: prefix_ids.size(0)] = DefaultToken.IGNORE_INDEX.value

            # ---------------- Mask padding ----------------
            ## If sequences were padded before (e.g., padding="max_length"), we must ignore
            # those padding tokens in the loss by setting their label positions to IGNORE_INDEX.

            ## If we use dynamic padding during collation (e.g. LLMDataCollator),
            # there are no padding tokens in input_ids at this point (padding is applied later),
            # so the line below will have no effect here (it must run after padding is added).
            labels[input_ids == pad_id] = DefaultToken.IGNORE_INDEX.value

            input_ids_list.append(input_ids)
            labels_list.append(labels)

            input_lens.append((input_ids != pad_id).sum().item())
            label_lens.append((labels != DefaultToken.IGNORE_INDEX.value).sum().item())

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "input_ids_lens": input_lens,
            "labels_lens": label_lens,
        }

    def preprocess(self, sources, targets):
        """
        Build chat messages and tokenize.
        """
        messages_list = []

        for src, tgt in zip(sources, targets):
            messages_list.append(
                [
                    {"role": "user", "content": src},
                    {"role": "assistant", "content": tgt},
                ]
            )

        tokenized = self._tokenize_fn(messages_list)
        
        # input_ids: what the model reads (input = full conversation)
        # labels: what the model is trained to predict (labels = only assistant tokens)
        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
        }

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "categories": self.categories[idx],
        }
