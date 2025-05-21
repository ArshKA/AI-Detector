import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
import config

logger = logging.getLogger(__name__)

def load_model_for_inference(adapter_path: str):
    logger.info(f"Loading PEFT adapter from: {adapter_path}")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    logger.info(f"Base model identified from adapter config: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer: pad_token set to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    quantization_config_inf = None
    if config.USE_4BIT_QUANTIZATION:
        quantization_config_inf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization for inference model loading.")
    logger.info(f"Loading base model '{base_model_name}' for inference...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        quantization_config=quantization_config_inf,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT_QUANTIZATION else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure the model's config also reflects the pad_token_id used by the tokenizer
    if tokenizer.pad_token_id is not None:
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Base Model: model.config.pad_token_id was None, explicitly set to {tokenizer.pad_token_id}")
        elif base_model.config.pad_token_id != tokenizer.pad_token_id:
            # If there's a mismatch, prioritize the tokenizer's pad_token_id as it's used for input preparation
            logger.warning(f"Base_model.config.pad_token_id ({base_model.config.pad_token_id}) differs from tokenizer.pad_token_id ({tokenizer.pad_token_id}). Overwriting model's config with tokenizer's pad_token_id.")
            base_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # This scenario implies an issue with the tokenizer's eos_token or its setup, which is unlikely with standard Hugging Face tokenizers but worth noting.
        logger.warning("tokenizer.pad_token_id is None after attempting to set pad_token. The model may still encounter issues with batch processing if padding is required.")

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info("PEFT model loaded and set to eval mode.")
    logger.info("PEFT model loaded and set to eval mode.")
    return model, tokenizer

def _list_jsonl_files(directory_path: str) -> list[str]:
    """Lists .jsonl files in a given directory."""
    logger.info(f"Looking for .jsonl files in directory: {directory_path}")
    if not directory_path.endswith('/'):
        directory_path += '/'
    try:
        jsonl_files = [f for f in os.listdir(directory_path) if f.endswith(".jsonl") and os.path.isfile(os.path.join(directory_path, f))]
        if not jsonl_files:
            logger.warning(f"No .jsonl files found in {directory_path}")
            return []
        logger.info(f"Found {len(jsonl_files)} .jsonl files: {jsonl_files}")
        return [os.path.join(directory_path, f) for f in jsonl_files]
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory_path}")
        return []
    except Exception as e:
        logger.error(f"Error listing files in {directory_path}: {e}")
        return []

def _load_texts_and_labels_from_file(filepath: str) -> tuple[list[str], list[int], int]:
    """Reads a .jsonl file, extracts texts and ground truth labels."""
    texts_to_predict = []
    ground_truths = []
    errors = 0
    filename = os.path.basename(filepath)
    logger.info(f"Reading and parsing file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"Processing {filename}")):
                try:
                    data = json.loads(line.strip())
                    human_text = data.get("human_text")
                    if human_text and isinstance(human_text, str):
                        texts_to_predict.append(human_text)
                        ground_truths.append(0) # 0 for human
                    machine_text = data.get("machine_text")
                    if machine_text and isinstance(machine_text, str):
                        texts_to_predict.append(machine_text)
                        ground_truths.append(1) # 1 for AI
                except json.JSONDecodeError as jde:
                    logger.warning(f"JSON decode error in {filename} at line {line_num+1}: {jde}. Line: '{line.strip()}'")
                    errors += 1
                except Exception as e: # Catch other potential errors during data extraction
                    logger.warning(f"Error processing line {line_num+1} in {filename}: {e}. Line: '{line.strip()}'")
                    errors +=1
    except FileNotFoundError:
        logger.error(f"File not found during load: {filepath}")
        # errors is already 0, texts and ground_truths are empty, so it will be skipped.
    except Exception as e:
        logger.error(f"Critical error reading file {filepath}: {e}")
        # Similar to FileNotFoundError, let it return empty lists and error count.
    
    if not texts_to_predict:
        logger.warning(f"No valid texts extracted from {filename}.")
    if errors > 0:
        logger.warning(f"Encountered {errors} errors while processing {filename}.")
    logger.info(f"Successfully loaded {len(texts_to_predict)} texts from {filename} with {errors} errors.")
    return texts_to_predict, ground_truths, errors

def _perform_batch_predictions(texts: list[str], model, tokenizer, batch_size: int) -> list[dict]:
    """Performs predictions on a list of texts in batches."""
    all_predictions = []
    logger.info(f"Starting batch prediction for {len(texts)} texts with batch size {batch_size}.")
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting batches"):
        batch_texts = texts[i:i + batch_size]
        # The actual prediction logic (tokenization, model call, postprocessing) is in the existing `predict` function.
        # We can call `predict` here for each batch.
        # Note: The existing `predict` function logs "Tokenizing X texts..." and "Performing inference...".
        # This might be a bit verbose if called for every small batch.
        # For this refactoring, I'll keep it as is, but a future optimization could be to make `predict` more granular
        # or have a separate internal batch prediction without per-batch logging.
        logger.debug(f"Predicting batch {i//batch_size + 1}/{ (len(texts) + batch_size - 1)//batch_size }")
        batch_predictions = predict(batch_texts, model, tokenizer) # predict already returns list of dicts
        all_predictions.extend(batch_predictions)
    logger.info(f"Batch prediction completed. Total predictions: {len(all_predictions)}.")
    return all_predictions

def _calculate_and_log_metrics(
    predictions: list[dict], 
    ground_truths: list[int], 
    filename: str, 
    file_loading_errors: int, 
    all_files_stats: dict, 
    overall_texts_processed: int, 
    overall_correct_predictions: int
) -> tuple[dict, int, int]:
    """Calculates, logs, and aggregates metrics for a single file."""
    
    file_stats = {
        "human_correct": 0, "human_total": 0,
        "ai_correct": 0, "ai_total": 0,
        "errors": file_loading_errors # Errors from loading the file
    }

    for i, pred_result in enumerate(predictions):
        predicted_label = pred_result["predicted_label"]
        # Ensure ground_truths has an entry for this prediction
        if i < len(ground_truths):
            ground_truth_label = ground_truths[i]
            overall_texts_processed += 1
            if predicted_label == ground_truth_label:
                overall_correct_predictions += 1

            if ground_truth_label == 0: # Human
                file_stats["human_total"] += 1
                all_files_stats["human_total"] += 1
                if predicted_label == 0:
                    file_stats["human_correct"] += 1
                    all_files_stats["human_correct"] += 1
            elif ground_truth_label == 1: # AI
                file_stats["ai_total"] += 1
                all_files_stats["ai_total"] += 1
                if predicted_label == 1:
                    file_stats["ai_correct"] += 1
                    all_files_stats["ai_correct"] += 1
        else:
            logger.warning(f"Prediction index {i} out of range for ground_truths with length {len(ground_truths)} in file {filename}. This prediction will be skipped in metrics calculation.")


    human_accuracy = (file_stats["human_correct"] / file_stats["human_total"] * 100) if file_stats["human_total"] > 0 else 0
    ai_accuracy = (file_stats["ai_correct"] / file_stats["ai_total"] * 100) if file_stats["ai_total"] > 0 else 0
    file_total_correct = file_stats["human_correct"] + file_stats["ai_correct"]
    file_total_samples = file_stats["human_total"] + file_stats["ai_total"]
    file_overall_accuracy = (file_total_correct / file_total_samples * 100) if file_total_samples > 0 else 0

    logger.info(f"Results for {filename}:")
    logger.info(f"  Human Texts: {file_stats['human_correct']}/{file_stats['human_total']} correct ({human_accuracy:.2f}%)")
    logger.info(f"  AI-Generated Texts: {file_stats['ai_correct']}/{file_stats['ai_total']} correct ({ai_accuracy:.2f}%)")
    logger.info(f"  Overall Accuracy for {filename}: {file_total_correct}/{file_total_samples} correct ({file_overall_accuracy:.2f}%)")
    if file_stats["errors"] > 0: # This now primarily refers to loading/parsing errors
        logger.warning(f"  ({file_stats['errors']} lines had errors during loading/parsing and were skipped)")
    
    return all_files_stats, overall_texts_processed, overall_correct_predictions

def _log_overall_summary(all_files_stats: dict, overall_texts_processed: int, overall_correct_predictions: int):
    """Logs the final evaluation summary."""
    logger.info("--- Overall Evaluation Summary ---")
    if overall_texts_processed == 0:
        logger.warning("No texts were processed in the evaluation. Cannot generate a summary.")
        return

    overall_human_accuracy = (all_files_stats["human_correct"] / all_files_stats["human_total"] * 100) if all_files_stats["human_total"] > 0 else 0
    overall_ai_accuracy = (all_files_stats["ai_correct"] / all_files_stats["ai_total"] * 100) if all_files_stats["ai_total"] > 0 else 0
    total_overall_accuracy = (overall_correct_predictions / overall_texts_processed * 100) # overall_texts_processed already checked
    
    logger.info(f"Total Human Texts Evaluated: {all_files_stats['human_total']}")
    logger.info(f"  Correct Human Predictions: {all_files_stats['human_correct']} ({overall_human_accuracy:.2f}%)")
    logger.info(f"Total AI-Generated Texts Evaluated: {all_files_stats['ai_total']}")
    logger.info(f"  Correct AI Predictions: {all_files_stats['ai_correct']} ({overall_ai_accuracy:.2f}%)")
    logger.info(f"Total Texts Processed: {overall_texts_processed}")
    logger.info(f"Total Correct Predictions: {overall_correct_predictions}")
    logger.info(f"Overall Accuracy: {total_overall_accuracy:.2f}%")

def predict(texts: list[str], model, tokenizer):
    # This function is now also used by _perform_batch_predictions.
    # Let's adjust logging slightly to avoid redundancy if called in a loop.
    # logger.info(f"Tokenizing {len(texts)} texts for inference...") # Potentially verbose in a loop
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH
    ).to(model.device)
    logger.info("Performing inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    scores_ai_generated = probabilities[:, 1].cpu().numpy()
    predicted_class_indices = torch.argmax(logits, dim=-1).cpu().numpy()
    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "predicted_label": int(predicted_class_indices[i]),
            "score_ai_generated": float(scores_ai_generated[i])
        })
    return results

def evaluate_on_dev_set(model, tokenizer, dev_data_path="cs162-final-dev/"):
    all_files_stats = {
        "human_correct": 0, "human_total": 0,
        "ai_correct": 0, "ai_total": 0
    }
    overall_texts_processed = 0
    overall_correct_predictions = 0
    logger.info(f"\n--- Starting Evaluation on Dev Set ({dev_data_path}) ---")
    
    jsonl_filepaths = _list_jsonl_files(dev_data_path)
    if not jsonl_filepaths:
        logger.warning("No .jsonl files found or directory error. Evaluation cannot proceed.")
        return

    for filepath in jsonl_filepaths:
        filename = os.path.basename(filepath)
        logger.info(f"--- Processing file: {filename} ---")
        
        texts_to_predict, ground_truths, file_loading_errors = _load_texts_and_labels_from_file(filepath)

        if not texts_to_predict:
            logger.warning(f"Skipping file {filename} due to no texts loaded or critical loading error.")
            # Errors are already logged by _load_texts_and_labels_from_file
            continue
        
        predictions = _perform_batch_predictions(texts_to_predict, model, tokenizer, config.INFERENCE_BATCH_SIZE)
        
        all_files_stats, overall_texts_processed, overall_correct_predictions = _calculate_and_log_metrics(
            predictions, ground_truths, filename, file_loading_errors,
            all_files_stats, overall_texts_processed, overall_correct_predictions
        )
            
    _log_overall_summary(all_files_stats, overall_texts_processed, overall_correct_predictions)

def main():
    parser = argparse.ArgumentParser(description="Run AI text detection inference or evaluation.")
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "evaluate"],
        help="Run mode: 'predict' for classifying texts or 'evaluate' for dev set evaluation."
    )
    parser.add_argument(
        "texts",
        nargs="*",
        type=str,
        help="One or more texts to classify (only if mode is 'predict')."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.INFERENCE_MODEL_PATH,
        help="Path to the fine-tuned LoRA adapter directory."
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default="cs162-final-dev/",
        help="Path to the directory containing .jsonl files for evaluation (only if mode is 'evaluate')."
    )
    args = parser.parse_args()
    if args.mode == "predict" and not args.texts:
        # parser.error already prints and exits, but good to log for completeness if it didn't.
        logger.error("The 'predict' mode requires at least one text to classify.")
        parser.error("The 'predict' mode requires at least one text to classify.")
    logger.info(f"Loading model from adapter path: {args.model_path}")
    model, tokenizer = load_model_for_inference(args.model_path)
    logger.info("Model and tokenizer loaded successfully.")
    if args.mode == "predict":
        predictions = predict(args.texts, model, tokenizer)
        logger.info("\n--- Inference Results ---")
        for res in predictions:
            label_str = "AI-Generated" if res["predicted_label"] == 1 else "Human-Written"
            log_message = (
                f"Text: \"{res['text'][:100]}...\"\n"
                f"  Prediction: {label_str} (Class {res['predicted_label']})\n"
                f"  Score (AI-Generated): {res['score_ai_generated']:.4f}\n"
                f"--------------------"
            )
            logger.info(log_message)
    elif args.mode == "evaluate":
        evaluate_on_dev_set(model, tokenizer, dev_data_path=args.dev_data_path)

if __name__ == "__main__":
    # Configure logging specifically for when the script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()

"""
python inference.py --mode predict --model_path saved_models/mistral_raid_detector_adapter/checkpoint-500 "I am a human" "I am an AI-generated text" "I'm a human" "This is human text lol"
python inference.py --mode evaluate --dev_data_path cs162-final-dev/ --model_path saved_models/mistral_raid_detector_adapter/checkpoint-500
"""