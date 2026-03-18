import sys
import logging
import torch
import transformers
import os

transformers.logging.set_verbosity(40)

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger

logger = logging.getLogger(__name__)


class FSChatBot(object):
    """
    A chatbot class that uses a language model for generating responses.

    This class implements a chatbot that can interact with users using natural
    language. It uses a pretrained language model as the backbone and can
    optionally load a fine-tuned checkpoint from federated learning. It can
    also use history and prompt templates to enhance the conversation quality.
    It provides two methods for generating responses: predict and generate.

    Attributes:
        tokenizer: A transformers.PreTrainedTokenizer object that can
            encode and decode text.
        model: A transformers.PreTrainedModel object that can generate text.
        device: A string representing the device to run the model on.
        add_special_tokens: A boolean indicating whether to add special tokens
            to the input and output texts.
        max_history_len: An integer representing the maximum number of
            previous turns to use as context.
        max_len: An integer representing the maximum number of tokens to
            generate for each response.
        history: A list of lists of integers representing the tokenized input
            and output texts of previous turns.
    """
    def __init__(self, config, model_to_eval="final", print_model_arch=False):
        """
        Initializes the chatbot with the given configuration.

        Args:
            config: A FS configuration object that contains various settings
                for the chatbot.
        """
        self.config = config
        model_name, model_hub = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len, model_hub)
        self.model = get_llm(config)

        self.device = f'cuda:{config.device}'
        self.add_special_tokens = True

        if config.llm.offsite_tuning.use:
            from federatedscope.llm.offsite_tuning.utils import \
                wrap_offsite_tuning_for_eval
            self.model = wrap_offsite_tuning_for_eval(self.model, config)
        else:
            try:
                original_path = config.federate.adapt_save_to
                dir_name = os.path.dirname(original_path)
                file_name = os.path.basename(original_path)

                if model_to_eval == "final":
                    specific_file_name = f"final_{file_name}"
                elif model_to_eval == "final_LDES":
                    specific_file_name = f"final_LDES_BEST_MODEL_{file_name}"
                else: #BEST MODEL of a client
                    specific_file_name = f"{model_to_eval}_BEST_MODEL_{file_name}"

                final_path = os.path.join(dir_name, specific_file_name)
                print("Checkpoint to use for evaluation:", final_path)
                ckpt = torch.load(final_path, map_location='cpu')
                if 'model' and 'cur_round' in ckpt:
                    self.model.load_state_dict(ckpt['model'])
                else:
                    self.model.load_state_dict(ckpt)
                if print_model_arch:
                    print("Model architecture:\n")
                    print(self.model)
                    print("\n")
            except Exception as error:
                print(f"{error}, will use raw model.")
                if print_model_arch:
                    print("Model architecture:\n")
                    print(self.model)
                    print("\n")
        if config.train.is_enable_half:
            self.model.half()

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        #if torch.__version__ >= "2" and sys.platform != "win32":
        #    self.model = torch.compile(self.model)

        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []
        
    def _build_prompt(self, input_text):
        """
        Builds a prompt template for the input text.

        Args:
            input_text: A string representing the user's input text.

        Returns:
            A string representing the source text with a prompt template.
        """
        source = {'instruction': input_text}
        # PROMPT = PROMPT_DICT_SUM if config.llm.chat.prompt_summary else PROMPT_DICT
        return PROMPT_DICT['prompt_no_input'].format_map(source)

    def predict(self, input_text, use_history=True, use_prompt=True):
        """
        Generates a response for the input text using the model.

        Args:
            input_text: A string representing the user's input text.
            use_history: A boolean indicating whether to use previous turns as
                context for generating the response. Default is True.
            use_prompt: A boolean indicating whether to use a prompt
                template for creating the source text. Default is True.

        Returns:
            A string representing the chatbot's response text.
        """
        if use_prompt:
            input_text = self._build_prompt(input_text)
        text_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = []
        if use_history:
            for history_ctx in self.history[-self.max_history_len:]:
                input_ids.extend(history_ctx)
        else:
            input_ids.extend(text_ids)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        response = self.model.generate(input_ids=input_ids,
                                       max_new_tokens=self.max_len,
                                       num_beams=4,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True,
                                       temperature=0.0)

        self.history.append(response[0].tolist())
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        return response_tokens

    @torch.no_grad()
    def generate(self, input_texts, generate_kwargs={}, chat_template=False, date_string=False):
        """
        Batched generation.

        Args:
            input_texts (str | List[str]): single prompt or list of prompts
            generate_kwargs (dict): HF generate kwargs
            chat_template (bool): whether to apply tokenizer chat template
            date_string (bool): whether to inject date_string

        Returns:
            responses (str | List[str])
            total_tokens_inp_out (int | List[int])
        """

        # ---------------------------
        # Normalize inputs to list
        # ---------------------------
        single_input = False
        if isinstance(input_texts, str):
            input_texts = [input_texts]
            single_input = True

        batch_size = len(input_texts)

        # ---------------------------
        # Apply chat template
        # ---------------------------
        if chat_template:
            processed_texts = []
            for text in input_texts:
                messages = [{"role": "user", "content": text}]
                if date_string:
                    from datetime import datetime
                    date_str = datetime.today().strftime("%Y-%m-%d")
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        date_string=date_str,
                    )
                else:
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                processed_texts.append(formatted)
        else:
            processed_texts = input_texts

        original_padding_side = self.tokenizer.padding_side
        use_left_padding = batch_size > 1 and original_padding_side != "left"
        if use_left_padding:
            self.tokenizer.padding_side = "left"

        try:
            model_inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(self.device)

            self.model.eval()
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                **generate_kwargs
            )
            responses = []
            token_counts = []
            prompt_padded_len = model_inputs.input_ids.shape[1]

            for i in range(batch_size):
                output_ids = generated_ids[i][prompt_padded_len:]
                responses.append(
                    self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        ignore_tokenization_space=True,
                    )
                )
                token_counts.append(generated_ids[i].shape[0])
        finally:
            if use_left_padding:
                self.tokenizer.padding_side = original_padding_side

        # ---------------------------
        # Return single or batch
        # ---------------------------
        if single_input:
            return responses[0], token_counts[0]

        return responses, token_counts


    def clear(self):
        """Clears the history of previous turns.

        This method can be used to reset the chatbot's state and start a new
        conversation.
        """
        self.history = []


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    chat_bot = FSChatBot(init_cfg)
    welcome = "Welcome to FSChatBot，" \
              "`clear` to clear history，" \
              "`quit` to end chat."
    print(welcome)
    while True:
        input_text = input("\nUser:")
        if input_text.strip() == "quit":
            break
        if input_text.strip() == "clear":
            chat_bot.clear()
            print(welcome)
            continue
        print(f'\nFSBot: {chat_bot.predict(input_text)}')


if __name__ == "__main__":
    main()
