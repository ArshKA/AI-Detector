import pandas as pd
import logging
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import config

logger = logging.getLogger(__name__)

def _load_raw_dataset(dataset_name: str, dataset_split: str) -> pd.DataFrame:
    """Loads data using load_dataset and converts to pandas DataFrame."""
    logger.info(f"Loading dataset: {dataset_name}, split: {dataset_split}")
    dataset = load_dataset(dataset_name, split=dataset_split)
    df = dataset.to_pandas()
    logger.info(f"Initial dataset size: {len(df)}")
    return df

def _preprocess_dataframe(df: pd.DataFrame, label_column: str, human_label_value: str, text_column: str) -> pd.DataFrame:
    """Handles label creation and ensures the text column is string type."""
    logger.info("Preprocessing dataframe...")
    df[label_column] = df['model'].apply(lambda x: 0 if x == human_label_value else 1)
    logger.info(f"Value counts for 'model' column:\n{df['model'].value_counts()}")
    logger.info(f"Value counts for '{label_column}' column:\n{df[label_column].value_counts()}")
    df[text_column] = df[text_column].astype(str).fillna('')
    logger.info("DataFrame preprocessing complete.")
    return df

def _subsample_dataframe(df: pd.DataFrame, max_samples: int, strategy: str, label_column: str, seed: int) -> pd.DataFrame:
    """Contains the logic for various subsampling strategies."""
    logger.info(f"Subsampling to {max_samples} samples using strategy: {strategy}")
    if strategy == "first_n":
        df_subsampled = df.head(max_samples)
    elif strategy == "random":
        df_subsampled = df.sample(n=max_samples, random_state=seed)
    elif strategy == "balanced_human_ai":
        human_df = df[df[label_column] == 0]
        ai_df = df[df[label_column] == 1]
        n_human = min(len(human_df), max_samples // 2)
        n_ai = min(len(ai_df), max_samples - n_human)
        if n_human < max_samples // 2 and len(ai_df) > n_ai: # Try to fill with AI if human samples are less
            n_ai = min(len(ai_df), max_samples - n_human)
        elif n_ai < (max_samples - (max_samples // 2)) and len(human_df) > n_human: # Try to fill with human if AI samples are less
            n_human = min(len(human_df), max_samples - n_ai)
        
        sampled_human_df = human_df.sample(n=n_human, random_state=seed)
        sampled_ai_df = ai_df.sample(n=n_ai, random_state=seed)
        df_subsampled = pd.concat([sampled_human_df, sampled_ai_df]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown subsampling strategy: {strategy}")
    logger.info(f"Subsampled dataset size: {len(df_subsampled)}")
    logger.info(f"Subsampled label distribution:\n{df_subsampled[label_column].value_counts()}")
    return df_subsampled

def _split_data(df: pd.DataFrame, test_split_size: float, seed: int, label_column: str = None) -> tuple[Dataset, Dataset | None]:
    """Splits data into training and validation Dataset objects."""
    if test_split_size > 0:
        stratify_col = df[label_column] if label_column and label_column in df.columns and df[label_column].nunique() > 1 else None
        train_df, val_df = train_test_split(
            df,
            test_size=test_split_size,
            random_state=seed,
            stratify=stratify_col
        )
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    else:
        train_dataset = Dataset.from_pandas(df)
        val_dataset = None
        logger.info(f"Train size: {len(train_dataset)}, No validation set.")
    return train_dataset, val_dataset

def _tokenize_datasets(train_dataset: Dataset, val_dataset: Dataset | None, tokenizer_name: str, text_column: str, label_column: str, max_length: int) -> tuple[Dataset, Dataset | None, AutoTokenizer]:
    """Handles tokenizer loading, dataset tokenization, and formatting."""
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer pad_token was None, set to eos_token.")

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    logger.info("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True) if val_dataset else None

    logger.info(f"Renaming column '{label_column}' to 'labels' for training dataset.")
    tokenized_train_dataset = tokenized_train_dataset.rename_column(label_column, "labels")
    logger.info("Setting format to 'torch' for training dataset.")
    tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    if tokenized_val_dataset:
        logger.info(f"Renaming column '{label_column}' to 'labels' for validation dataset.")
        tokenized_val_dataset = tokenized_val_dataset.rename_column(label_column, "labels")
        logger.info("Setting format to 'torch' for validation dataset.")
        tokenized_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    logger.info("Tokenization and formatting complete.")
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

def load_and_preprocess_data(tokenizer_name: str) -> tuple[Dataset, Dataset | None, AutoTokenizer]:
    df = _load_raw_dataset(config.DATASET_NAME, config.DATASET_SPLIT)
    df = _preprocess_dataframe(df, config.LABEL_COLUMN, config.HUMAN_LABEL_VALUE, config.TEXT_COLUMN)
    if config.MAX_SAMPLES is not None and config.MAX_SAMPLES < len(df):
        df = _subsample_dataframe(df, config.MAX_SAMPLES, config.SUBSAMPLE_STRATEGY, config.LABEL_COLUMN, config.SEED)
    
    train_dataset, val_dataset = _split_data(df, config.TEST_SPLIT_SIZE, config.SEED, config.LABEL_COLUMN)
    
    tokenized_train_dataset, tokenized_val_dataset, tokenizer = _tokenize_datasets(
        train_dataset, val_dataset, tokenizer_name, config.TEXT_COLUMN, config.LABEL_COLUMN, config.MAX_LENGTH
    )
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

if __name__ == '__main__':
    # Configure logging specifically for when the script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing data_utils.py...")
    # Note: config.MODEL_NAME should be a valid tokenizer name for this test.
    # If it's a path, ensure the tokenizer files are accessible.
    tokenizer_name_for_test = config.MODEL_NAME 
    
    logger.info(f"Using tokenizer: {tokenizer_name_for_test} for the test run.")

    train_data, val_data, tokenizer_instance = load_and_preprocess_data(tokenizer_name_for_test)
    
    logger.info(f"Training data sample: {train_data[0]}")
    if val_data:
        logger.info(f"Validation data sample: {val_data[0]}")
    else:
        logger.info("No validation data was created (as per configuration).")
        
    logger.info(f"Tokenizer pad token ID: {tokenizer_instance.pad_token_id}, EOS token ID: {tokenizer_instance.eos_token_id}")
    logger.info("Data loading and preprocessing test in __main__ completed successfully.")
    logger.info("\nSample from training data:")
    logger.info(train_data[0])
    if val_data:
        logger.info("\nSample from validation data:")
        logger.info(val_data[0])
    logger.info(f"\nTokenizer pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
    logger.info("Data loading and preprocessing test complete.")