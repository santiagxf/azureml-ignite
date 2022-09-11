"""
This modules provides consistent preprocessing for all the text used in models at training
and inference type.
"""
import os
import glob
import math
from typing import List

import pandas as pd

def split_to_sequences(text: str, unique_words, seq_len) -> List[str]:
    """
    Splits a text sequence of an arbitrary length to sub-sequences of no more than `seq_len`
    words. Each sub-sequence will have `unique_words` words from the original text and the
    remaining words to match `seq_len` will be brought from the previous sub-sequence. This
    way we are able to retain some of the context from the previous sequence while splitting text.

    Parameters
    ----------
    text : str
        Text you want to split in multiple subsequences.
    unique_words : int
        Number of unique words to use on each subsequence.
    seq_len : int
        Number of total words to output on each sequence. Each sequence would then contain
        `seq_len - unique_words` from the previous subsequence (context) and then `unique_words`
        from the current subsequence being generated.

    Returns
    -------
    List[str]
        A list of sub-sequences of text of no more than `sequence_len`.
    """
    assert unique_words<seq_len

    words = text.split()
    n_seq = math.ceil(len(words)/unique_words)

    seqs = [' '.join(words[seq*unique_words:seq*unique_words + seq_len]) for seq in range(n_seq)]
    return seqs

def split_sentences(input_dataset: str, text_column_name:str, split_sentences: bool, 
                    max_words_sentence: int, words_carry_on: int, output_dataset:str):

    if os.path.isdir(input_dataset):
        input_dataset = os.path.join(input_dataset, "*.csv")

    if not str(input_dataset).endswith('.csv'):
        raise TypeError('Only CSV files are supported by the loading data procedure.')

    if not glob.glob(input_dataset):
        raise FileNotFoundError(f"Path or directory {input_dataset} doesn't exists")

    df = pd.concat(map(pd.read_csv, glob.glob(input_dataset)))
    if split_sentences:
        df.loc[:,text_column_name] = df[text_column_name].apply(split_to_sequences,
                                            unique_words=(max_words_sentence - words_carry_on),
                                            seq_len=max_words_sentence).explode(text_column_name).reset_index(drop=True)

    
    df.to_csv(os.path.join(output_dataset, "data.csv"), index=False)
