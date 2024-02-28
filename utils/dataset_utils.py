# utils/dataset_utils.py
from datasets import load_dataset

def load_and_prepare_dataset(dataset_name, split='train'):
    """
    Load and prepare a dataset for training or evaluation.

    Args:
    - dataset_name: The name or path of the dataset to load.
    - split: The specific split of the dataset to load, e.g., 'train', 'validation', 'test'.

    Returns:
    - A loaded and optionally preprocessed dataset ready for training or evaluation.
    """
    # Load the dataset from Hugging Face Hub
    dataset = load_dataset(dataset_name, split=split)

    
    return dataset
