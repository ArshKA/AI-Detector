import torch
import logging
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import config

logger = logging.getLogger(__name__)

def get_model(model_name: str, num_labels: int = 2):
    logger.info(f"Loading base model: {model_name}")
    quantization_config = None
    if config.USE_4BIT_QUANTIZATION:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization (QLoRA).")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=quantization_config,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT_QUANTIZATION else config.BNB_4BIT_COMPUTE_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        logger.info(f"Set model.config.pad_token_id to EOS token ID: {model.config.eos_token_id}")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    logger.info("PEFT LoRA model configured.")
    # model.print_trainable_parameters() # This prints to stdout, consider capturing and logging
    # Capture the output of print_trainable_parameters()
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    model.print_trainable_parameters()
    sys.stdout = old_stdout
    logger.info(f"Trainable parameters: {captured_output.getvalue().strip()}")
    return model

if __name__ == '__main__':
    # Configure logging specifically for when the script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing model_utils.py...")
    model = get_model(config.MODEL_NAME)
    logger.info("Model architecture:") # Removed newline, logger typically handles it or it's part of the model string
    logger.info(model) # This will print the model structure, which can be very verbose
    logger.info("Model configuration:") # Removed newline
    logger.info(model.config) # This will print the model config, also verbose
    logger.info("Model loading test complete.")