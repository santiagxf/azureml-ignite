import logging
import mlflow
import os
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import DataType
from datasets.arrow_dataset import Dataset, Features, Value
from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, default_data_collator

eval_metric = load_metric('accuracy')

def load_raw_dataset(train_path, validation_path, text_column_name, 
                     label_column_name, cache_dir=".cache"):

    if os.path.isdir(train_path):
        print('is path')
        train_path = os.path.join(train_path, "*.csv")
    if os.path.isdir(validation_path):
        print('is path')
        validation_path = os.path.join(validation_path, "*.csv")

    data_files = {}
    data_files["train"] = train_path
    data_files["validation"] = validation_path

    feature_mappings = {label_column_name: "int32", text_column_name: "string"}
    features = (
            Features({feature: Value(dtype) for feature, dtype in feature_mappings.items()})
        )
    
    return Dataset.from_csv(data_files, features=features, cache_dir=cache_dir)


def tokenize_and_batch_datasets(tokenizer, raw_datasets, text_column_name):
    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding="max_length", max_length=128, truncation=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return eval_metric.compute(predictions=predictions, references=labels)


def finetune(weights_path: str, tokenizer_path: str, config_path: str, 
             train_path: str, validation_path: str, 
             text_column_name: str, label_column_name: str, num_labels: int,
             batch_size: int, num_train_epochs: int, model_output: str,
             weights_output: str, tokenizer_output: str, config_output: str,
             ort: bool = False, fp16: bool = False, deepspeed: bool = False):

    if validation_path == None:
        logging.warning("[WARN] No evaluation dataset has been provided. Using training data for evaluation")
        validation_path = train_dataset

    logging.info("[DEBUG] Reading datasets from inputs")
    raw_datasets = load_raw_dataset(train_path, validation_path, text_column_name, label_column_name)

    logging.info("[DEBUG] Loading base model, config and tokenizer")
    config = AutoConfig.from_pretrained(config_path)
    config.num_labels = num_labels

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(weights_path, config=config)
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    train_dataset, eval_dataset = tokenize_and_batch_datasets(
        tokenizer, raw_datasets, text_column_name
    )

    training_args_dict = {
        'output_dir': './results',
        'logging_dir': './logs', 
        'report_to': 'none',
        'num_train_epochs': num_train_epochs,
        "per_device_train_batch_size" : batch_size,
        "evaluation_strategy": "epoch",
    }

    log_model = True
    if ort or fp16:
        logging.info('[DEBUG] Enabling ORT module for training')
        training_args_dict["ort"] = ort
        training_args_dict["fp16"] = fp16
    if deepspeed:
        logging.info('[DEBUG] Enabling deepspeed configuration with paramters ds_config_zero_1.json')
        training_args_dict["deepspeed"] = "ds_config_zero_1.json"
        if 'RANK' in os.environ.keys():
            rank = os.environ['RANK']
            log_model = rank == 0
            logging.info(f"[DEBUG] RANK = {rank}")

    training_args = TrainingArguments(**training_args_dict)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    history = trainer.train()
    evaluation_metrics = trainer.evaluate()

    mlflow.log_metrics(dict(filter(lambda item: item[1] is not None, evaluation_metrics.items())))
    mlflow.log_params(training_args_dict)

    tokenizer.save_pretrained(tokenizer_output)
    model.save_pretrained(weights_output)
    config.save_pretrained(config_output)

    if log_model:
        logging.info('[DEBUG] Logging MLflow model')
        tokenizer.save_pretrained(model_output)
        model.save_pretrained(model_output)

        signature = ModelSignature(
            inputs=Schema([
                ColSpec(DataType.string, text_column_name),
            ]),
            outputs=Schema([
                ColSpec(DataType.integer, "rating"),
                ColSpec(DataType.double, "confidence"),
            ]))

        mlflow.pyfunc.log_model('model', 
                                data_path=model_output,
                                code_path=["./hg_loader_module.py"], 
                                loader_module="hg_loader_module", 
                                signature=signature)
