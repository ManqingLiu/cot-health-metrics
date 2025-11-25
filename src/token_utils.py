import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import ModelConfig

class TokenUtils:
    def __init__(self, hf_model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def decode_to_string(self, tokens: torch.Tensor, skip_special_tokens=True):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens)

    def encode_to_tensor(self, string: str, to_device=None):
        if to_device is None:
            to_device = self.hf_model.device
        return self.tokenizer.encode(
            string, return_tensors="pt").to(to_device)

    def escape_string(self, string: str):
        return string.encode('unicode_escape').decode()

    ## extract end_think based on model name, if Qwen is used, use </think>, else use fuzzy end think list from model config
    def _get_end_think_token(self, model):
        model_name = model.model_name
        model_config = ModelConfig.get(model_name)
        if "end_think" in model_config:
            end_think = model_config["end_think"]
        elif "fuzzy_end_think_list" in model_config:
            # Use the first token in the fuzzy_end_think_list as end_think
            end_think = model_config["fuzzy_end_think_list"][0]
        else:
            print(f"ERROR: model {model_name} missing CoT separator config")
            exit(1)
        return end_think

    def get_answer_log_probs_recalc(self, model, prompt: str, cot: str, prediction: str):
        """ Get log probs for just the answer (prediction), given prompt+cot+prediction.
        
            Note: prompt should end with a <think> token if required.
            cot should not contain <think> or </think>.
        """
        # Determine end_think token
        end_think = self._get_end_think_token(model)
        if cot == "":
            text0 = prompt + end_think
        else:
            text0 = prompt + cot + end_think
        text = text0 + prediction

        text0_tokens = self.encode_to_tensor(text0)
        text_tokens = self.encode_to_tensor(text)

        log_probs = model.get_log_probs(text_tokens)

        skip_count = text0_tokens.shape[1] #- 1  # -1 for EOS token
        return self.get_token_log_probs(log_probs, text_tokens, skip_count)

    def get_answer_log_probs_recalc_no_cot(self, model, prompt_no_cot: str, prediction: str):
        """ Get log probs for just the answer (prediction), given prompt_no_cot+prediction.
        """
        text = prompt_no_cot + prediction

        text0_tokens = self.encode_to_tensor(prompt_no_cot)
        text_tokens = self.encode_to_tensor(text)

        print(f"prompt_no_cot: {prompt_no_cot}")
        print(f"text: {text}")

        log_probs = model.get_log_probs(text_tokens)

        skip_count = text0_tokens.shape[1]
        return self.get_token_log_probs(log_probs, text_tokens, skip_count)

    def get_answer_log_probs_recalc_batch(self, model, prompt: list[str], cot: list[str], prediction: list[str]):
        """ Get log probs for just the answer (prediction), given prompt+cot+prediction.
        
            Note: prompt should end with a <think> token if required.
        """
        end_think = self._get_end_think_token(model)
        text_tokens_list = []
        skip_count_list = []
        for (prompt, cot, prediction) in zip(prompt, cot, prediction):
            if cot == "":
                text0 = prompt + end_think
            else:
                text0 = prompt + cot + end_think
            text = text0 + prediction

            text0_tokens = self.encode_to_tensor(text0)
            text_tokens = self.encode_to_tensor(text)
            text_tokens_list.append(text_tokens)
            skip_count = text0_tokens.shape[1] #- 1  # -1 for EOS token
            skip_count_list.append(skip_count)

        log_probs_list = model.get_log_probs_batch(text_tokens_list)

        token_log_probs_list = [self.get_token_log_probs(log_probs, text_tokens, skip_count)
            for log_probs, text_tokens, skip_count in zip(log_probs_list, text_tokens_list, skip_count_list)]
        return token_log_probs_list

    def get_token_log_probs(self, log_probs, tokens, start_index=0):
        """Get probabilities for tokens from [start_index,end)."""
        batch_size, seq_len, vocab_size = log_probs.shape
        end_index = min(seq_len, tokens.shape[1])

        # Ensure start_index is valid (>= 0)
        if start_index == 0 or end_index == 0:
            raise ValueError("start_index is 0, maybe there is no CoT?")

        # go back one index in logits to get next token probability for each token
        actual_tokens = tokens[0, start_index:end_index]
        
        # Extract the relevant slice of log probabilities (shifted back by 1)
        log_probs_slice = log_probs[0, start_index-1:end_index-1]
        
        # Reshape actual_tokens for gather operation (add dimension for gathering)
        actual_tokens_reshaped = actual_tokens.unsqueeze(-1)
        
        # Gather the log probabilities for the actual tokens
        gathered_log_probs = log_probs_slice.gather(1, actual_tokens_reshaped)
        
        # Remove the extra dimension
        token_log_probs = gathered_log_probs.squeeze(-1)

        return token_log_probs
