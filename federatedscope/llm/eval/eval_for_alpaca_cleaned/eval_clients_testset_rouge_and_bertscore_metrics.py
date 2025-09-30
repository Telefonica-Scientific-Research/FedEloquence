import re
import os
import transformers
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot
import importlib
import bert_score

transformers.logging.set_verbosity(40)

# Evaluate ROUGE-L score only
def evaluate_rouge_l(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores['rougeL'].fmeasure

# Evaluate BERTScore with multilingual model
def evaluate_bertscore(reference, generated, device, lang="en"):
    _, _, F1 = bert_score.score([generated], [reference], lang=lang, model_type="xlm-roberta-large", device=device)
    return F1.mean().item()

# Build prompt for model input
def build_prompt(instruction, input, domain, prompt):
    domain_prompts = prompt.get(domain, {})
    formatted_source = (
        domain_prompts.get("prompt_input", "").format(instruction=instruction, input=input) 
        if input not in ("", None) 
        else domain_prompts.get("prompt_no_input", "").format(instruction=instruction)
    )
    return formatted_source

# Main function adapted for alpaca_cleaned_8c_clients_testset.jsonl
def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)
    device = init_cfg.device

    path_to_prompt = init_cfg.llm.prompt_path
    module_path = path_to_prompt.replace("/", ".")
    module = importlib.import_module(module_path)
    PROMPT = module.PROMPT

    # Initialize chatbot with the fine-tuned adapter
    fschatbot = FSChatBot(init_cfg, args.model_to_eval)

    # Load test set
    test_file = f'{init_cfg.data.root}/alpaca_cleaned_8c_clients_testset.jsonl'
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found.")
        return
    list_data_dict = load_jsonl(test_file, instruction='instruction', input='input', output='output', domain='domain')

    # Initialize dicts
    scores_by_lang = {lang: {"rougeL": [], "bertscore": []}
                      for lang in ["en","ca","es","de","da","eu","hr","sr"]}

    # Evaluate each sample
    for idx, sample in enumerate(tqdm(list_data_dict)):
        input_text = build_prompt(sample['instruction'], sample['input'], sample['domain'], PROMPT)
        generate_kwargs = dict(max_new_tokens=init_cfg.llm.chat.max_len) 
        model_completion = fschatbot.generate(input_text, generate_kwargs)

        print(input_text)
        print(model_completion + "\n")  
        reference = sample["output"]
        print(f'### Reference:\n{reference}\n')

        # ROUGE-L
        rouge_l = evaluate_rouge_l(reference, model_completion)

        # BERTScore (per-language if supported, else fallback to "en")
        try:
            F1 = evaluate_bertscore(reference, model_completion, device, lang=sample["domain"])
        except:
            F1 = evaluate_bertscore(reference, model_completion, device, lang="en")

        lang = sample["domain"]
        if lang in scores_by_lang:
            scores_by_lang[lang]["rougeL"].append(rouge_l)
            scores_by_lang[lang]["bertscore"].append(F1)

        print(f"ROUGE-L: {rouge_l:.3f}")
        print(f"BERTScore-F1: {F1:.3f}")
        print("--------------------------------------------------------")

    # Print per-language averages
    for lang, scores in scores_by_lang.items():
        if len(scores["rougeL"]) > 0:
            print(f'Final scores for {lang.upper()}:')
            print(f'ROUGE-L: {np.mean(scores["rougeL"]):.3f}')
            print(f'BERTScore-F1: {np.mean(scores["bertscore"]):.3f}')
            print("--------------------------------------------------------")

if __name__ == "__main__":
    main()