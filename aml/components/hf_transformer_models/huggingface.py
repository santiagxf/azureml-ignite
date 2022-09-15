import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModel


def get_model(model_name: str, config_output: str, tokenizer_output: str, weights_output: str):
    config = AutoConfig.from_pretrained(model_name)
    out_config = Path(config_output).resolve().absolute()
    print(f"Saving configuration to {out_config}")
    config.save_pretrained(str(out_config))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    out_tokenizer = Path(tokenizer_output).resolve().absolute()
    print(f"Saving tokenizer to {out_tokenizer}")
    tokenizer.save_pretrained(str(out_tokenizer))

    model = AutoModel.from_pretrained(model_name)
    out_weights = Path(weights_output).resolve().absolute()
    print(f"Saving weights to {out_weights}")
    model.save_pretrained(str(out_weights))
