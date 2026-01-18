import json
import random
from typing import Dict, List, Tuple
from config import ICL_EXAMPLES_DIRECTORY_DEFAULT


class ICLExampleGenerator:
    """Generator for in-context learning examples with different filler types."""

    def __init__(self):
        # Base questions and answers for math problems
        self.base_examples = [
            ("What is 45 ÷ 5?", "9"),
            ("A rectangle has length 10 and width 6. What is its perimeter?", "32"),
            ("What is 17 + 28?", "45"),
            ("If you save $25 per week for 4 weeks, how much will you have?", "$100"),
            ("What is 12 × 12?", "144"),
            ("A book costs $15. If you buy 3 books, how much do you spend?", "$45"),
            ("What is 100 - 37?", "63"),
            ("What is 64 ÷ 8?", "8"),
            ("A train has 6 cars with 20 seats each. How many total seats?", "120"),
            ("What is 33 - 17?", "16"),
            ("If eggs cost $3 per dozen, how much do 3 dozen cost?", "$9"),
            ("What is the value of 5³ (5 cubed)?", "125"),
            ("What is 8 × 7?", "56"),
            ("If you have 50 marbles and lose 18, how many are left?", "32"),
            ("What is 144 ÷ 12?", "12")
        ]

        # Lorem ipsum words for generating filler text
        self.lorem_words = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
            "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
            "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
            "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
            "consequat", "duis", "aute", "irure", "reprehenderit", "in", "voluptate",
            "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
            "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
            "deserunt", "mollit", "anim", "id", "est", "laborum"
        ]

    def get_think_tokens(self, model_type: str = "default") -> Tuple[str, str]:
        """Get the appropriate think tokens based on model type.

        For gpt-oss-20b, uses harmony format:
        - Analysis channel: <|start|>assistant<|channel|>analysis<|message|>...<|end|>
        - Final channel: <|start|>assistant<|channel|>final<|message|>...<|end|><|return|>
        """
        if model_type == "gpt-oss-20b":
            return ("<|start|>assistant<|channel|>analysis<|message|>",
                    "<|end|>")
        else:
            return ("<think>", "</think>")

    def generate_think_filler(self, length: int = 50) -> str:
        """Generate filler text with repeated 'think' words."""
        return " ".join(["think"] * length)

    def generate_analysis_filler(self, length: int = 50) -> str:
        """Generate filler text with repeated 'analyze' words."""
        return " ".join(["analysis"] * length)

    def generate_dot_filler(self, length: int = 100) -> str:
        """Generate filler text with dots."""
        return "." * length

    def generate_comma_filler(self, length: int = 100) -> str:
        """Generate filler text with commas."""
        return "," * length

    def generate_lorem_filler(self, word_count: int = 20) -> str:
        """Generate lorem ipsum filler text."""
        return " ".join(random.choices(self.lorem_words, k=word_count))

    def generate_mixed_filler(self, length: int = 50) -> str:
        """Generate mixed filler with various characters."""
        chars = [".", ",", "think", "analyze", "consider"]
        return " ".join(random.choices(chars, k=length))

    def create_icl_example(self, question: str, answer: str, filler_content: str,
                           model_type: str = "default") -> Dict:
        """Create a single ICL example with the specified format."""
        begin_think, end_think = self.get_think_tokens(model_type)

        return {
            "question": question,
            "cot": f"{begin_think} {filler_content} {end_think}",
            "answer": answer
        }

    def generate_dataset(self, filler_type: str, num_examples: int = 5,
                         model_type: str = "default", length = 20) -> Dict:
        """Generate a complete dataset with specified filler type."""
        # Select random examples
        selected_examples = random.sample(self.base_examples, min(num_examples, len(self.base_examples)))

        examples = []
        for question, answer in selected_examples:
            # Generate appropriate filler based on type
            if filler_type == "think_token":
                filler = self.generate_think_filler(length=length)
            elif filler_type == "analysis":
                filler = self.generate_analysis_filler(length=length)
            elif filler_type == "dot":
                filler = self.generate_dot_filler(length=length)
            elif filler_type == "comma":
                filler = self.generate_comma_filler(length=length)
            elif filler_type == "lorem_ipsum":
                filler = self.generate_lorem_filler(word_count=length)
            elif filler_type == "mixed":
                filler = self.generate_mixed_filler(length=length)
            else:
                raise ValueError(f"Unknown filler type: {filler_type}")

            example = self.create_icl_example(question, answer, filler, model_type)
            examples.append(example)

        return {filler_type: examples}

    def save_dataset(self, dataset: Dict, filename: str):
        """Save dataset to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved dataset to {filename}")

    def generate_all_datasets(self, num_examples: int = 5, model_type: str = "default", directory: str = ICL_EXAMPLES_DIRECTORY_DEFAULT, length = 20):
        """Generate and save all filler type datasets."""
        filler_types = ["think_token", "dot", "comma", "lorem_ipsum", "mixed", "analysis"]

        for filler_type in filler_types:
            dataset = self.generate_dataset(filler_type, num_examples, model_type, length)
            filename = f"{directory}/icl_{filler_type}_{model_type}_{num_examples}_fewshot.json"
            self.save_dataset(dataset, filename)

    def generate_model_specific_datasets(self, num_examples: int = 5, directory: str = ICL_EXAMPLES_DIRECTORY_DEFAULT):
        """Generate datasets for different model types."""
        model_types = ["default", "gpt-oss-20b"]
        filler_types = ["think_token", "dot", "lorem_ipsum", "analysis"]

        for model_type in model_types:
            for filler_type in filler_types:
                dataset = self.generate_dataset(filler_type, num_examples, model_type)
                filename = f"{directory}/icl_{filler_type}_{model_type}.json"
                self.save_dataset(dataset, filename)


def main():
    """Main function to demonstrate usage."""
    generator = ICLExampleGenerator()

    # Generate all datasets with default think tokens
    # print("Generating datasets with default think tokens...")
    # generator.generate_all_datasets(num_examples=5, model_type="gpt-oss-20b")
    generator.generate_all_datasets(num_examples=5, model_type="default", length=20)

    # Generate model-specific datasets
    # print("\nGenerating model-specific datasets...")
    # generator.generate_model_specific_datasets(num_examples=5)

    # Example: Generate a custom dataset
    # print("\nGenerating custom example...")
    # custom_dataset = generator.generate_dataset("analysis", num_examples=5, model_type="gpt-oss-20b")
    # print(json.dumps(custom_dataset, indent=2))


if __name__ == "__main__":
    main()