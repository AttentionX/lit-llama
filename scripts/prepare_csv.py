import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.tokenizer import Tokenizer

import requests
import io
import PyPDF2
import numpy as np
import os
import json
from tqdm import tqdm
import csv

import openai_api
import paths
import prepare_arxiv

TOKENIZER_PATH = paths.TOKENIZER_PATH
TOKENIZED_DATA_PATH = Path('data/tokenized')
QA_DATASET_PATH = Path('data/qa_dataset')

IGNORE_INDEX = -1

def read_papers_from_csv(csv_path:str='data/notable/2023.txt'):
    papers = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            print(i+1, 'th file in', csv_path.split('/')[-1])
            if len(line) >= 3:
                url = line[2]
                papers.append(url)
                prepare_arxiv.retrieveArxiv(url)
            i += 1
    return papers

def main():
    read_papers_from_csv()

if __name__ == "__main__":
    main()