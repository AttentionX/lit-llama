import sys
from pathlib import Path
import json

import openai_api

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

TOKENIZER_PATH = '/Users/adamlee/Downloads/AttentionX/models/llama/tokenizer/tokenizer.model'

def prepare(path:Path):
    with open(path, "r") as file:
        data = json.load(file)
    instruction = """
    Create a list of 3 paraphrases for the following sentence that can be given as an alternative to ansewr the following question:
    """