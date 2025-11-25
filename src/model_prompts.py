from src.config import ModelConfig

class ModelPromptBuilder:
    """ Creates a single instance of a prompt builder. Do not reuse this class.

        Please call the following:
        - add messages with one of the following:
            - add_to_history() with role="system" or role="user"
            - add_partial_to_history() if needed
            - add_user_message() for default behavior to format a question
        - add_cot_mode() which may insert a think token or a no-think sequence
        - make_prompt()
    """

    def __init__(self, model_name: str, invokes_cot: bool = True, invokes_filler: bool = False):
        self.model_name = model_name
        self.invokes_cot = invokes_cot
        self.invokes_filler = invokes_filler
        self.question = None

        # default to making a new assistant role section
        self.continue_final_message = False
        self.add_generation_prompt = True
        self.history = []
        self.append_after_apply = ""

    def add_to_history(self, role: str, content: str):
        assert self.continue_final_message == False

        self.history.append({
            "role": role,
            "content": content
        })

    def add_partial_to_history(self, role: str, content: str):
        assert self.continue_final_message == False

        self.history.append({
            "role": role,
            "content": content
        })
        self.continue_final_message = True
        self.add_generation_prompt = False

    def _get_model_custom_instruction(self):
        """ Used for model-specific prompts.  """

        please_write_answer = "Please state reasoning first and then write the string \"Answer: \" before the final answer."

        if self.model_name.startswith(("google/", "meta-llama/", "mistralai/")):
            return please_write_answer
        return None

    def _get_user_message_components(self, question: str, custom_instruction: str = None):
        instructions = []
        instructions.append(f"Question: {question}")

        if custom_instruction is None:
            # Only add "Let's think step by step" if CoT is enabled
            if self.invokes_cot and not self.invokes_filler:
                instructions.append("Let's think step by step.")
        elif custom_instruction.strip():  # Only add if not empty
            instructions.append(custom_instruction)

        model_custom_instruction = self._get_model_custom_instruction()
        if model_custom_instruction is not None:
            instructions.append(model_custom_instruction)

        if self.invokes_cot == False:
            # Strong, explicit anti-think instructions for non-CoT mode with ICL examples
            anti_think_instruction = """IMPORTANT: Give ONLY the final answer. Do NOT show your work. Do NOT explain your reasoning. Do NOT use any tags. Just state the answer directly."""

            if "gpt-oss" in self.model_name:
                anti_think_instruction += " <|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant"
            else:
                anti_think_instruction += " /no_think"
            instructions.append(anti_think_instruction)
        else:
            # Very strong, explicit anti-think instructions after answer tag for CoT mode
            anti_think_instruction = "IMPORTANT: After you finish reasoning, state the final answer directly after \"Answer:\". DO NOT include REASONING steps after the final answer."
            instructions.append(anti_think_instruction)

        return instructions

    def add_user_message(self, question: str, custom_instruction: str = None, ground_truth_answer: str = None):
        self.question = question

        instructions = self._get_user_message_components(question, custom_instruction)

        message = "\n".join(instructions)
        self.add_to_history("user", message)

    def _add_think_token(self):
        model_config = ModelConfig.get(self.model_name)
        if "begin_think" in model_config:
            if (self.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
                self.add_partial_to_history("assistant", "<think>")
            elif (self.model_name == "openai/gpt-oss-20b"):
                self.add_partial_to_history("assistant", "analysis")
        elif "fuzzy_end_think_list" in model_config:
            pass
        else:
            print(f"ERROR: model {self.model_name} missing CoT separator config")
            exit(1)

    def add_cot_mode(self):
        if self.invokes_cot:
            self._add_think_token()

    def make_prompt(self, tokenizer):
        prompt = self._apply_chat_template(tokenizer)
        prompt += self.append_after_apply
        return prompt

    def _apply_chat_template(self, tokenizer):
        # Use enable_thinking parameter to control think tags at the tokenizer level
        enable_thinking = self.invokes_cot

        try:
            # Try to use enable_thinking parameter if supported
            prompt = tokenizer.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
                continue_final_message=self.continue_final_message,
                enable_thinking=enable_thinking
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking parameter
            prompt = tokenizer.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
                continue_final_message=self.continue_final_message
            )

        return prompt


class CustomInstructionPromptBuilder(ModelPromptBuilder):
    def __init__(self, model_name: str, custom_instruction: str, custom_assistant_prefix: str = "",
                 invokes_cot: bool = True, invokes_filler: bool = False):
        super().__init__(model_name, invokes_cot, invokes_filler)
        self.custom_instruction = custom_instruction
        self.custom_assistant_prefix = custom_assistant_prefix

    def add_user_message(self, question: str, custom_instruction_: str = None, ground_truth_answer: str = None):
        # If we have custom_instruction_, use it; otherwise use empty string to prevent default "Let's think step by step"
        passed_instruction = custom_instruction_ if custom_instruction_ is not None else ""

        instructions = self._get_user_message_components(question, passed_instruction)

        # Insert the organism's custom instruction at the beginning
        if self.custom_instruction:
            instructions.insert(0, self.custom_instruction)

        message = "\n".join(instructions)
        self.add_to_history("user", message)

    def add_partial_to_history(self, role: str, content: str):
        content += self.custom_assistant_prefix
        super().add_partial_to_history(role, content)


class AddGroundTruthPromptBuilder(CustomInstructionPromptBuilder):
    def __init__(self, model_name: str, custom_instruction: str = " The answer is ",
                 invokes_cot: bool = True, invokes_filler: bool = False):
        super().__init__(model_name, invokes_cot, invokes_filler)
        self.custom_instruction = custom_instruction

    def add_user_message(self, question: str, custom_instruction_: str = None, ground_truth_answer: str = None):
        if ground_truth_answer is None:
            raise ValueError("ground_truth_answer is required for AddGroundTruthPromptBuilder")
        question = question + " The answer is " + ground_truth_answer
        #print(f"AddGroundTruthPromptBuilder: question: {question}, custom_instruction_: {custom_instruction_}")
        return super().add_user_message(question, custom_instruction_)


class ICLPromptBuilder(CustomInstructionPromptBuilder):
    """Prompt builder that includes in-context learning examples"""

    def __init__(self, model_name: str, custom_instruction: str,
                 custom_assistant_prefix: str = "", icl_examples: list = None,
                 filler_type: str = "think", invokes_cot: bool = True):
        super().__init__(model_name, custom_instruction, custom_assistant_prefix, invokes_cot)
        self.icl_examples = icl_examples or []
        self.filler_type = filler_type

    def _get_user_message_components(self, question: str, custom_instruction_: str = None):
        instructions = super()._get_user_message_components(question, custom_instruction_)
        return instructions

    def add_user_message(self, question: str, custom_instruction: str = None, ground_truth_answer: str = None):
        """Override to include ICL examples before the main question"""
        self.question = question

        # Build the complete prompt with ICL examples
        prompt_parts = []

        # Add ICL examples with simple numbering
        if self.icl_examples:
            for i, example in enumerate(self.icl_examples, 1):
                # Format each example as "Example N: Question: ... <think>...</think> Answer: ..."
                prompt_parts.append(
                    f"\nExample {i}:\nQuestion: {example['question']}\nCoT: {example['cot']}\nAnswer: {example['answer']}")

        # Add instruction about solving pattern
        prompt_parts.append(
            "Now solve the following question using the same pattern. Then end thinking mode and output your final answer, with no extra reasoning steps.")

        # Add the question without any default "Let's think step by step"
        prompt_parts.append(f"Question: {question}")

        # Add organism's custom instruction if it exists
        if self.custom_instruction:
            prompt_parts.append(self.custom_instruction)

        # Add model-specific instructions
        model_custom_instruction = self._get_model_custom_instruction()
        if model_custom_instruction is not None:
            prompt_parts.append(model_custom_instruction)

        # Join all parts
        full_prompt = "\n".join(prompt_parts)
        self.add_to_history("user", full_prompt)
