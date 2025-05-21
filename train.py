import torch
import numpy as np
import logging
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import config
from data_utils import load_and_preprocess_data
from model_utils import get_model

logger = logging.getLogger(__name__)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)
    logger.info(f"CUDA available. Set seed for CUDA to {config.SEED}")
else:
    logger.info("CUDA not available.")


def compute_metrics(pred):
    labels = pred.label_ids
    preds_logits = pred.predictions
    preds_indices = np.argmax(preds_logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_indices, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds_indices)

    probs = torch.softmax(torch.tensor(preds_logits), dim=-1)[:, 1].numpy()
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.5

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }

def main():
    logger.info("Starting training process...")
    logger.info(f"Loading data for model: {config.MODEL_NAME}")
    train_dataset, val_dataset, tokenizer = load_and_preprocess_data(tokenizer_name=config.MODEL_NAME)

    if train_dataset is None or len(train_dataset) == 0:
        logger.warning("No training data loaded. Exiting.")
        return

    logger.info("Loading model...")
    model = get_model(model_name=config.MODEL_NAME, num_labels=2)

    logger.info("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        optim=config.OPTIMIZER,
        logging_dir=config.LOGGING_DIR,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="roc_auc" if val_dataset else None,
        greater_is_better=True if val_dataset else None,
        report_to="wandb" if "wandb" in config.LOGGING_DIR else "tensorboard",
        fp16=config.BNB_4BIT_COMPUTE_DTYPE == torch.float16,
        bf16=config.BNB_4BIT_COMPUTE_DTYPE == torch.bfloat16,
        seed=config.SEED,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    callbacks = []
    if val_dataset:
        logger.info("Early stopping enabled.")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001))

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    logger.info(f"Saving LoRA adapter model to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)

    metrics = train_result.metrics
    logger.info(f"Training metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Training metrics, state, and model saved.")

    if val_dataset:
        logger.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info(f"Validation metrics: {eval_metrics}")
    else:
        logger.info("No validation set. Skipping evaluation.")

    logger.info(f"Model and tokenizer adapter saved to {config.OUTPUT_DIR}")
    logger.info("Training script completed.")

if __name__ == "__main__":
    # Configure logging specifically for when the script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
