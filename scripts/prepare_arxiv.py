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
import torch
import copy
import math

import openai_api
import paths

TOKENIZER_PATH = paths.TOKENIZER_PATH
TOKENIZED_DATA_PATH = Path('data/tokenized')
QA_DATA_PATH = Path('data/qa')
QA_DATASET_PATH = Path('data/qa_dataset')

MAX_PAGES = 30

# 1: Tokenize
# 2: QA
# 3: QA Dataset
TYPE = 3

IGNORE_INDEX = -1

urls = [
    'https://arxiv.org/pdf/2303.08774.pdf',
]

def cut_references(paper:str):
    ref_index = paper.find('References\n')
    if ref_index > 0:
        paper = paper[:ref_index]
    return paper

def cut_references_pages(pages:list):
    final_pages = [paper if "References\n" not in paper else paper[:paper.find('References\n')] for paper in pages]
    return final_pages
    final_papers = []
    for paper in papers:
        if "References\n" not in paper:
            final_papers.append(paper)
        else:
            ref_index = paper.find('References\n')
            paper = paper[:ref_index]
            final_papers.append(paper)
    return final_papers

def retrieve_pdf_from_url(url:str):
    # Download the PDF file from the URL
    response = requests.get(url)

    # Extract the content of the PDF file as bytes
    content = io.BytesIO(response.content)

    # Read the PDF file using PyPDF2
    pdf_reader = PyPDF2.PdfReader(content)

    # Extract the text from the PDF file
    pages = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pages.append(page.extract_text())
        if page_num == MAX_PAGES:
            break
    
    return cut_references_pages(pages)

def prepare_paraphrases(dataset_path):
    qa_dataset_path = f"{dataset_path}/original_train.jsonl"
    destination_path = f"{dataset_path}/train.jsonl"
    test_destination_path = f"{dataset_path}/test.jsonl"
    jsonl = []
    test_jsonl = []
    k = 4

    with open(qa_dataset_path, 'r') as jsonl_file:
        total_lines = sum(1 for line in jsonl_file)
    
    i = 0
    with open(qa_dataset_path, 'r') as jsonl_file:
        for line in jsonl_file:
            # print(line)
            if line[-2] != '}' and line[-1] != '}':
                print('skipping', line)
                print(f'"{line[-1]}" "{line[-2]}"')
                continue
            json_obj = json.loads(line)
            paraphrased_questions = get_paraphrased_questions(json_obj['question'], json_obj['answer'], k=k)
            # print(paraphrased_questions)

            j = 0
            for paraphrased_question in paraphrased_questions:
                new_json_obj = copy.deepcopy(json_obj)
                new_json_obj['original_question'] = new_json_obj['question']
                new_json_obj['question'] = paraphrased_question
                print('Question:', new_json_obj['question'])
                if j == k-1:
                    # Save to test josnl
                    test_jsonl.append(new_json_obj)
                else:
                    # Save to train jsonl
                    jsonl.append(new_json_obj)
                j += 1
            i += 1
            print(f'{i}/{total_lines}')
    
    print('Saving train jsonl')
    mode = 'w'
    if os.path.exists(destination_path):
        mode = 'a'
    with open(destination_path, mode) as f:
        for obj in jsonl:
            f.write(json.dumps(obj) + '\n')

    print('Saving test jsonl')
    mode = 'w'
    if os.path.exists(test_destination_path):
        mode = 'a'
    with open(test_destination_path, mode) as f:
        for obj in test_jsonl:
            f.write(json.dumps(obj) + '\n')

def get_paraphrased_questions(question, answer, k=3):
    instruction = f"""
    Refer to the example above,
    Based on the following question, create {k} distinct paraphrased questions, which are different from each other, that are semantically equivalent to the original question such that when asked any of the unique paraphrased questions you will generate the same response.
    Question: {question}
    Response: {answer}
    Generate {k} Paraphrased Questions (separated by newlines):\n
    """
    examples = """
    Question: <Question>
    Response: <Response>
    Generate 3 Paraphrased Questions:
    1. <Question1>
    2. <Question2>
    3. <Question3>
    """
    prompt = f"{examples}\n{instruction}"
    results = openai_api.chatGPT(prompt)

    print(results)

    return [result[3:].strip() for result in results.split('\n')]

def get_paraphrased_question(question, answer):
    instruction = f"""
    Based on the following question, create a paraphrased question that is semantically equivalent to the original question such that when asked the paraphrased question you will generate the same response.
    Question: {question}
    Response: {answer}
    Paraphrased Question: 
    """
    return openai_api.chatGPT(instruction)

def prepare_validation(qa_dataset_path):
    destination_path = qa_dataset_path.replace('train', 'test')
    jsonl = []
    i = 0
    with open(qa_dataset_path, 'r') as jsonl_file:
        for line in jsonl_file:
            # print(line)
            if line[-2] != '}' and line[-1] != '}':
                print('skipping', line)
                print(f'"{line[-1]}" "{line[-2]}"')
                continue
            json_obj = json.loads(line)
            paraphrased_question = get_paraphrased_question(json_obj['question'], json_obj['answer'])
            if paraphrased_question is False:
                continue
            json_obj['original_question'] = json_obj['question']
            json_obj['question'] = paraphrased_question
            jsonl.append(json_obj)
            i += 1
            print(f'{i}')
    mode = 'w'
    if os.path.exists(destination_path):
        mode = 'a'
    with open(destination_path, mode) as f:
        for obj in jsonl:
            # if obj[-1] != '}':
            #     continue
            # data = json.loads(obj)
            f.write(json.dumps(obj) + '\n')
            

# Prepare a dataset with the qa jsonl file
def prepareQADataset(file_path, destination_path:Path, max_seq_length: int = 256, mask_inputs_for_label: bool = True, add_prior_prompt=False):
    tokenizer = Tokenizer(TOKENIZER_PATH)

    # Fix reading jsonl file
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    dataset = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs_for_label, add_prior_prompt) for sample in tqdm(data)]
    torch.save(dataset, destination_path)

def prepare_sample(sample, tokenizer, mask_inputs, add_prior_prompt=False, max_seq_length=256):
    prior_prompt = """Write a response that appropriately completes the question."""
    question = sample['question']
    answer = sample['answer']
    if add_prior_prompt is False:
        prior_prompt = ''
    encoded_question = tokenizer.encode(f"{prior_prompt}{question}")
    encoded_full = tokenizer.encode(f'{prior_prompt}{question} {answer}')
    labels = encoded_full.clone()
    if mask_inputs:
        labels[:len(encoded_question)] = IGNORE_INDEX
    return {**sample, "input_ids": encoded_full, "input_ids_no_response": encoded_question, "labels": labels}
    

# Prepare dataset in q&a format
def prepareQA(text, destination_path:Path, title, date=None, chunk_size=7000, skip_if_exists=True):
    if os.path.exists(destination_path / f'{title}.jsonl') and skip_if_exists:
        return

    qa_examples = """
    {"question": "When did Virgin Australia start operating?", "answer": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route."}
    {"question": "When was Tomoaki Komorida born?", "answer": "Tomoaki Komorida was born on July 10,1981."}
    {"question": "Who was Kyle Van Zyl playing against when he scored 36 of hisa teams 61 points?", "answer": "Kyle Van Zyl was playing against Boland U21 when he scored 36 points, leading his team to victory in a 61-3 win."}
    """
    instruction = """
    Given the following information, create as many question-answer pairs as possible about the information, covering all of the core topics/subjects in jsonl format (Each line containing a json object). The questions should be phrased such that you would be able to answer it if asked and should contain enough context. 
    """
    i = 0
    jsonl = []

    # Switch chunks to be paragraphs that overlap

    while i * chunk_size < len(text):
        print(f'Processing chunk {i+1} in {title} dataset of length {math.ceil(len(text)/chunk_size)}')
        end_index = (i+1)*chunk_size if (i+1)*chunk_size < len(text) else len(text)
        chunk = text[i*chunk_size:end_index]
        
        full_title = text.split('\n')[0]
        context = f'The given information is an excerpt from the paper titled {full_title}'
        if date is not None:
            year = f'20{date[:2]}'
            month = date[2:4]
            context += f' published in {month} {year}.\n'
        
        instruction = f'{context}\n{instruction}'

        prompt = f"{instruction}\n\nExample Q&A:\n{qa_examples}\n\nInformation:\n{chunk}\n\nQ&A:"
        answer = openai_api.chatGPT(prompt)
        # print(answer)
        answer = answer.split('\n')
        jsonl.extend(answer)
        i += 1
        # if i == 5:
        #     break
    
    print(f'Saving {title} dataset of qa pairs {len(jsonl)}')
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
    """Prepare the dataset in "Tiny Shakespeare" style (pre-training)."""
    print(f"Preparing the arxiv dataset ... {len(text)} characters")
    title = text.split('\n')[0]

    # Tokenizer.train(input=input_file_path, destination=destination_path, vocab_size=100)
    tokenizer = Tokenizer(TOKENIZER_PATH)
    text_ids = tokenizer.encode(text)
    print(f"{title} has {len(text_ids):,} tokens")
    text_ids = np.array(text_ids, dtype=np.uint16)
    text_ids.tofile(destination_path / f"{title}_raw.bin")

def getTitle(string):
    if ':' in string:
        title = string.split(':')[0]
    elif ' ' in string:
        words = string.split(' ')
        if len(words) > 2:
            title = ' '.join(words[:2])
    else:
        title = string
    if len(title) > 20:
            title = title[:20]
    return title

def getDate(arxivUrl):
    index = arxivUrl.find('pdf')
    return arxivUrl[index+4:index+8]
    

def retrieveArxiv(url, option=1):
    date = getDate(url)

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
        if page_num == MAX_PAGES:
            break
    
    title = getTitle(text.split('\n')[0])
    file_title = date + "_" + title
    full_title = text.split('\n')[0]
    print(f'Saving {title}')
    text_path = f"data/raw_text/{file_title}.txt"
    if not os.path.exists(text_path):
        with open(text_path, "w") as file:
            file.write(text)

    # Print the extracted text
    # print(text)
    print(f'Saved {title}')
    if option == 1:
        prepareQA(text, QA_DATA_PATH, file_title)
    elif option == 2:
        parseArxiv(text, TOKENIZED_DATA_PATH)
    # elif option == 3:
    #     prepareQADataset(QA_DATA_PATH / title / 'test.jsonl', QA_DATASET_PATH / title / 'test.pt')

if __name__ == '__main__':
    prepare_paraphrases('data/qa/GPT-4')
    
    # prepareQADataset(QA_DATA_PATH / 'GPT-4' / 'test.jsonl', QA_DATASET_PATH / 'GPT-4' / 'test.pt')

    # prepare_validation('data/qa/GPT-4/train.jsonl')

    # for url in urls:
    #     retrieveArxiv(url, TYPE)
