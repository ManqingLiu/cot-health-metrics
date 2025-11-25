import json


def clean_answer(answer_text):
    """Clean up answer text by removing extra markers"""
    if not answer_text:
        return ""

    # Remove <|im_end|> and "Answer:" prefix
    cleaned = answer_text.replace("<|im_end|>", "").strip()
    if cleaned.startswith("Answer:"):
        cleaned = cleaned[7:].strip()

    return cleaned


def convert_log_to_json(input_file, output_file):
    result = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                log_entry = json.loads(line)

                transformed_entry = {
                    "question": log_entry.get("question", ""),
                    "cot": log_entry.get("cot", ""),
                    "answer": clean_answer(log_entry.get("answer", ""))  # Clean the answer
                }

                result.append(transformed_entry)

            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(result)} entries to {output_file}")


if __name__ == "__main__":
    convert_log_to_json("data/custom/binary_alternation.log", "data/custom/binary_alternation.json")