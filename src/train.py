import transformers
import datasets
import mlflow
from datasets.arrow_dataset import Dataset, Features, Value
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

def create_dataset(data_path:str):
    feature_mappings = {"label": "int32", "text": "string"}
    features = (
            Features({feature: Value(dtype) for feature, dtype in feature_mappings.items()})
        )
    
    return Dataset.from_csv(data_path, features=features)

def fine_tune(train_path:str, eval_path:str, baseline:str):
    train_input = create_dataset(train_path)
    eval_input = create_dataset(eval_path)

    tokenizer = AutoTokenizer.from_pretrained(baseline)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(baseline, num_labels=5)
    model.config.pad_token_id = model.config.eos_token_id

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", max_length=128, truncation=True)

    train_dataset = train_input.map(tokenize_function, batched=True)
    eval_dataset = eval_input.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        logging_dir='./logs', 
        report_to = 'none',
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    history = trainer.train()
    evaluation_metrics = trainer.evaluate()

    mlflow.log_metrics(dict(filter(lambda item: item[1] is not None, evaluation_metrics.items())))
    mlflow.log_params(history.metrics)
