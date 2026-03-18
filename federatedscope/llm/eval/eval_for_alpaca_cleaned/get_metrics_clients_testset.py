import os
import json
import importlib
import requests
import numpy as np
from tqdm import tqdm
import re
import torch

import transformers
from rouge_score import rouge_scorer
import bert_score

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot


transformers.logging.set_verbosity(40)

# =====================================================
# Metrics
# =====================================================

ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def evaluate_rouge_l(reference, generated):
    return ROUGE_SCORER.score(reference, generated)["rougeL"].fmeasure

def evaluate_bertscore(reference, generated, device, lang):
    _, _, f1 = bert_score.score(
        [generated],
        [reference],
        lang=lang,
        model_type="xlm-roberta-large",
        device=device,
        rescale_with_baseline=False
    )
    return f1.mean().item()


# =====================================================
# Prompt construction
# =====================================================

def build_prompt(instruction, input_text, language, prompt_dict):
    language_prompt = prompt_dict.get(language)
    if language_prompt is None:
        raise ValueError(f"No prompt template for language: {language}")

    if input_text not in ("", None):
        return language_prompt["prompt_input"].format(
            instruction=instruction,
            input=input_text,
        )
    else:
        return language_prompt["prompt_no_input"].format(
            instruction=instruction
        )


# =====================================================
# LLM-as-a-Judge
# =====================================================

def query_llama_server(prompt, max_tokens=1):
    try:
        r = requests.post(
            "http://127.0.0.1:10002/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1.0,
                "top_k": 1,
                "max_tokens": max_tokens,
                "seed": 42,
            },
            timeout=100,
        )
        if r.status_code != 200:
            print("Judge HTTP error:", r.text)
            return ""
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Judge server exception:", e)
        return ""


def evaluate_llm_judge(judge_prompt):
    output = query_llama_server(judge_prompt)
    try:
        score = int(output[0])
        return score if 1 <= score <= 5 else None
    except Exception:
        return None


def build_judge_prompt(judge_prompt_dict, sample, reference, generated):
    lang = sample["language"]
    prompts = judge_prompt_dict.get(lang)
    if prompts is None:
        raise ValueError(f"No judge prompt for language: {lang}")

    args = {
        "instruction": sample["instruction"],
        "input": sample.get("input", ""),
        "reference": reference,
        "generated": generated,
    }

    if sample.get("input") not in ("", None):
        return prompts["prompt_input"].format_map(args)
    else:
        return prompts["prompt_no_input"].format_map(args)


# =====================================================
# Utility
# =====================================================

def chunked(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# =====================================================
# Main
# =====================================================

def main():
    torch.backends.cudnn.deterministic = False # True
    torch.backends.cudnn.benchmark = True # False

    # ---------- FederatedScope args ----------
    fs_args = parse_args()
    init_cfg = global_cfg.clone()

    if fs_args.cfg_file:
        init_cfg.merge_from_file(fs_args.cfg_file)

    cfg_opt, client_cfg_opt = parse_client_cfg(fs_args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    device = init_cfg.device

    # ---------- Output ----------
    os.makedirs(os.path.dirname(fs_args.output_path), exist_ok=True)
    metrics_txt_path = fs_args.output_path.replace(".jsonl", "_metrics.txt")

    # ---------- Load prompts ----------
    prompt_module = importlib.import_module(
        init_cfg.llm.prompt_path.replace("/", ".")
    )
    PROMPT = prompt_module.PROMPT

    testset_match = re.match(
        r"^(.*?)(_clients_testset)(?:_.*)?$",
        (fs_args.testset_name or "").strip(),
    )
    if testset_match is None:
        raise ValueError(
            "Invalid --testset_name. Expected a name containing "
            "'_clients_testset' (optionally followed by extra suffixes), "
            "e.g. 'alpaca_cleaned_8c_clients_testset_with_DA'."
        )
    beginning_testset_name = testset_match.group(1)
    judge_module = importlib.import_module(
        f"prompt_templates.{beginning_testset_name}_llm_as_a_judge"
    )
    JUDGE_PROMPT = judge_module.JUDGE_PROMPT

    # ---------- Load data ----------
    test_file = f"{init_cfg.data.root}/{fs_args.testset_name}.jsonl"
    samples = load_jsonl(
        test_file,
        instruction="instruction",
        input="input",
        output="output",
        language="language",
    )

    # ---------- Model ----------
    chatbot = FSChatBot(init_cfg, fs_args.model_to_eval)

    generate_kwargs = {
        "max_new_tokens": init_cfg.llm.chat.max_len,
        "temperature": 0,
        "do_sample": False
    }

    # ---------- Metrics storage ----------
    scores_by_lang = {}

    # ---------- Evaluation ----------
    global_idx = 0
    with open(fs_args.output_path, "w", encoding="utf-8") as fout:
        for batch in tqdm(chunked(samples, fs_args.batch_size)):

            prompts = [
                build_prompt(
                    s["instruction"],
                    s.get("input"),
                    s["language"],
                    PROMPT,
                )
                for s in batch
            ]

            generations, _ = chatbot.generate(
                prompts,
                generate_kwargs=generate_kwargs,
                chat_template=True,
                date_string=True,
            )
            if len(generations) != len(batch):
                raise RuntimeError(
                    f"Generation size mismatch: got {len(generations)} outputs "
                    f"for batch size {len(batch)}."
                )

            for sample, generated in zip(batch, generations):
                reference = sample["output"]
                lang = sample["language"]

                if lang not in scores_by_lang:
                    scores_by_lang[lang] = {
                        "rougeL": [],
                        "bertscore_f1": [],
                        "llm_judge_score": [],
                    }

                metrics = {}

                if fs_args.use_rouge:
                    metrics["rougeL"] = evaluate_rouge_l(reference, generated)
                    scores_by_lang[lang]["rougeL"].append(metrics["rougeL"])

                if fs_args.use_bertscore:
                    try:
                        metrics["bertscore_f1"] = evaluate_bertscore(
                            reference, generated, device, lang
                        )
                    except Exception:
                        metrics["bertscore_f1"] = evaluate_bertscore(
                            reference, generated, device, "en"
                        )
                    scores_by_lang[lang]["bertscore_f1"].append(metrics["bertscore_f1"])

                if fs_args.use_llm_judge:
                    judge_prompt = build_judge_prompt(
                        JUDGE_PROMPT,
                        sample,
                        reference,
                        generated,
                    )
                    metrics["llm_judge_score"] = evaluate_llm_judge(judge_prompt)
                    scores_by_lang[lang]["llm_judge_score"].append(metrics["llm_judge_score"])

                record = {
                    "sample_id": global_idx,
                    "instruction": sample["instruction"],
                    "input": sample.get("input", ""),
                    "reference": reference,
                    "generated": generated,
                    "lang": lang,
                    "metrics": metrics,
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_idx += 1

    # ---------- Write per-language metrics ----------
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        for lang, scores in scores_by_lang.items():
            f.write(f"Language: {lang}\n")

            if scores["rougeL"]:
                f.write(f"  ROUGE-L: {np.mean(scores['rougeL']):.4f}\n")

            if scores["bertscore_f1"]:
                f.write(f"  BERTScore-F1: {np.mean(scores['bertscore_f1']):.4f}\n")

            judge_vals = [v for v in scores["llm_judge_score"] if v is not None]
            total = len(scores["llm_judge_score"])
            valid = len(judge_vals)

            if valid > 0:
                f.write(
                    f"  LLM-as-a-Judge: {np.mean(judge_vals):.4f} "
                    f"(n={valid}/{total})\n"
                )
            else:
                f.write("  LLM-as-a-Judge: N/A (n=0)\n")

            f.write("-" * 50 + "\n")

    print("Evaluation completed successfully.")
    print(f"Saved JSONL to: {fs_args.output_path}")
    print(f"Saved language metrics to: {metrics_txt_path}")


if __name__ == "__main__":
    main()
