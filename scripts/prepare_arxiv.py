import requests
import io
import PyPDF2
import numpy as np
import os
import json
import sys
from pathlib import Path

import openai_api

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

TOKENIZER_PATH = '/Users/adamlee/Downloads/AttentionX/models/llama/tokenizer/tokenizer.model'
TOKENIZED_DATA_PATH = Path('../data/tokenized')
QA_DATA_PATH = Path('../data/qa')

# 1: Tokenize
# 2: QA
TYPE = 2

urls = [
    'https://arxiv.org/pdf/2303.08774.pdf',
]

# Prepare dataset in q&a format
def prepareQA(text, destination_path:Path, title):
    qa_examples = """
    {"question": "When did Virgin Australia start operating?", "answer": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route."}
    {"question": "When was Tomoaki Komorida born?", "answer": "Tomoaki Komorida was born on July 10,1981."}
    {"question": "Regarding Lollapalooza, where does it take place, who started it and what is it?", "answer": "Lollapalooze is an annual musical festival held in Grant Park in Chicago, Illinois. It was started in 1991 as a farewell tour by Perry Farrell, singe of the group Jane's Addiction. The festival includes an array of musical genres including alternative rock, heavy metal, punk rock, hip hop, and electronic dance music. The festivals welcomes an estimated 400,000 people each year and sells out annually. Some notable headliners include: the Red Hot Chili Peppers, Chance the Rapper, Metallica, and Lady Gage. Lollapalooza is one of the largest and most iconic festivals in the world and a staple of Chicago."}
    {"question": "Who was Kyle Van Zyl playing against when he scored 36 of hisa teams 61 points?", "answer": "Kyle Van Zyl was playing against Boland U21 when he scored 36 points, leading his team to victory in a 61-3 win."}
    """
    instruction = """
    Given the following information, create as many question-answer pairs as possible about the information, covering all of the core topics/subjects in jsonl format(Each line containing a json object).
    """
    i = 0
    jsonl = []
    while i * 10000 < len(text):
        end_index = (i+1)*10000 if (i+1)*10000 < len(text) else len(text)
        chunk = text[i*10000:end_index]
        prompt = f"{instruction}\n\nExample Q&A:\n{qa_examples}\n\nInformation:\n{chunk}\n\nQ&A:"
        answer = openai_api.chatGPT(prompt)
        print(answer)
        answer = answer.split('\n')
        jsonl.extend(answer)
        i += 1
        # if i == 5:
        #     break
    
    # Save answer to jsonl file, switch to append mode 'a' if file already exists
    mode = 'w'
    if os.path.exists(destination_path / f'{title}.jsonl'):
        mode = 'a'
    with open(destination_path / f'{title}.jsonl', mode) as f:
        for obj in jsonl:
            if obj[-1] != '}':
                continue
            data = json.loads(obj)
            f.write(json.dumps(data) + '\n')

# Prepare dataset for arxiv articles
def parseArxiv(text, destination_path: Path = Path("data/alpaca")):
    """Prepare the "Tiny Shakespeare" dataset."""
    from lit_llama import Tokenizer
    print(f"Preparing the arxiv dataset ... {len(text)} characters")
    title = text.split('\n')[0]

    # Tokenizer.train(input=input_file_path, destination=destination_path, vocab_size=100)
    tokenizer = Tokenizer(TOKENIZER_PATH)
    text_ids = tokenizer.encode(text)
    print(f"{title} has {len(text_ids):,} tokens")
    text_ids = np.array(text_ids, dtype=np.uint16)
    text_ids.tofile(destination_path / f"{title}_raw.bin")

def retrieveArxiv(url, option=1):
    # Download the PDF file from the URL
    response = requests.get(url)

    # Extract the content of the PDF file as bytes
    content = io.BytesIO(response.content)

    # Read the PDF file using PyPDF2
    pdf_reader = PyPDF2.PdfReader(content)

    # Extract the text from the PDF file
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
        if page_num == 30:
            break
    
    title = text.split('\n')[0]
    text_path = f"../data/raw_text/{title}.txt"
    if not os.path.exists(text_path):
        with open(text_path, "w") as file:
            file.write(text)

    # Print the extracted text
    print(text)
    if option == 1:
        parseArxiv(text, TOKENIZED_DATA_PATH)
    elif option == 2:
        prepareQA(text, QA_DATA_PATH, title)

if __name__ == '__main__':
    for url in urls:
        retrieveArxiv(url, TYPE)