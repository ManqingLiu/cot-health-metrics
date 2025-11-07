import reasoning_gym
import json
import os
import argparse


def generate_dataset(dataset_name: str, sample_size: int, output_path: str) -> None:
    """
    Generate maze dataset and save to JSON file.

    Args:
        dataset_name: Dataset name
        sample_size: Number of samples to generate
        output_path: Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get the dataset
    data = reasoning_gym.create_dataset(dataset_name,
		#min_grid_size=200,
		#max_grid_size=200,
		#min_dist=200,
		#max_dist=1000,
		size=sample_size, seed=42)

    # Create a list to store all entries
    dataset = []
    for x in data:
        entry = {
            'question': x['question'],
            'answer': x['answer'],
            'metadata': x['metadata']
        }
        dataset.append(entry)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return len(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate maze dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--max_animals', type=int, default=20, required=False)
    parser.add_argument('--output_path', type=str, default=f'data/maze_dataset.json',
                        help='Path to save the JSON file')

    args = parser.parse_args()

    count = generate_dataset(args.dataset_name, args.sample_size, args.output_path)
    print(f"{args.dataset_name} dataset with {count} entries saved to {args.output_path}")

