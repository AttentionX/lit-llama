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
import re

import openai_api
import paths
import prepare_arxiv

TOKENIZER_PATH = paths.TOKENIZER_PATH
TOKENIZED_DATA_PATH = Path('data/tokenized')
QA_DATASET_PATH = Path('data/qa_dataset')

IGNORE_INDEX = -1

CHARS_PER_SECTION = 300

def generate_questions(paper_info:list, text:str, page_num:int, reference:str=None):
    context = f'The following is from the paper "{paper_info[0]}" released in {paper_info[1]}'
    reference_instruction = f'Here is the summary of the abstract of the paper for reference:\n"{reference}"' if reference else ''
    # Mention the paper title and page num in the json
    examples = '{"question": "When did Virgin Australia start operating?", "answer": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route."}\n{"question": "When was Tomoaki Komorida born?", "answer": "Tomoaki Komorida was born on July 10,1981."}\n{"question": "Who was Kyle Van Zyl playing against when he scored 36 of hisa teams 61 points?", "answer": "Kyle Van Zyl was playing against Boland U21 when he scored 36 points, leading his team to victory in a 61-3 win."}'
    instruction = f'Given the information about the paper above, generate as many question-answer pairs as possible, in the following format:\n{examples}\n about the following section from the paper in page {page_num}, covering all of the core topics/subjects that are crucial to the essence of the paper and its significance in jsonl format (Each line containing a json object).'
    question_specific_instruction = f'The questions should be phrased generally, such that you would be able to answer it independently, without knowing the paper. So questions like "What is figure 4?" or "What is the name of the paper?" are unacceptable since they aren\'t general and can\'t be answered independently'
    prompts = [context]
    if page_num != 0:
        prompts.append(reference_instruction)
    prompts.append(f"{instruction}\n{question_specific_instruction}")
    prompts.append(f'Page {page_num}:\n{text}')
    prompt = '\n\n'.join(prompts)
    # prompt = f'{context}\n\n{reference_instruction}\n\n{instruction}\n\n{question_specific_instruction}\n\nPage {page_num}:\n{text}'
    answer = openai_api.chatGPT(prompt)
    return answer

def get_page_text(sections_of_pages:list, page_num:int):
    cur_page_sections = sections_of_pages[page_num]
    if page_num != 0:
        # Append last section of previous page to very front of the current page
        cur_page_sections.insert(0, sections_of_pages[page_num-1][-1])
    if page_num != len(sections_of_pages) - 1:
        # Append first section of next page to very end of the current page
        cur_page_sections.append(sections_of_pages[page_num+1][0])
    curr_page_text = '\n\n'.join(cur_page_sections)
    return curr_page_text

def get_sections(page:str):
    final_sections = []

    sections = page.split('.\n')
    sections = [re.sub('\s+', ' ', section).strip() for section in sections]

    skip = []
    i = 0
    j = 0
    for section in sections:
        if i in skip:
            i += 1
            continue
        
        while j < len(sections) - 1 and len(section) < CHARS_PER_SECTION:
            j += 1
            section += '\n' + sections[j]
            skip.append(j)
        final_sections.append(section)
        i += 1
        if i > j:
            j = i
    return final_sections

def read_papers_from_csv(csv_path:str='data/notable/adam.txt'):
    destination_path = Path('data/qa/notable')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            print(i+1, 'th file in', csv_path.split('/')[-1])
            if len(line) >= 4:
                url = line[3]
                date = line[1]
                title = line[0]
                
                clean_title = title.replace(' ', '_')
                out_title = f'{i}_{clean_title}.jsonl'
                
                pages = prepare_arxiv.retrieve_pdf_from_url(url)
                pages_sections = [get_sections(page) for page in pages]
                beginning_of_paper = pages_sections[0][:2]
                for page in pages_sections:
                    page_text = get_page_text(pages_sections, pages_sections.index(page))
                    qas = generate_questions([title, date], page_text, pages_sections.index(page), reference='\n'.join(beginning_of_paper))
                    qas = qas.split('\n')
                    qas = [json.loads(qa) for qa in qas if '}' in qa]
                    # Write qas to json
                    mode = 'w'
                    if os.path.exists(destination_path / f'{out_title}.jsonl'):
                        mode = 'a'
                    with open(destination_path / f'{out_title}.jsonl', mode) as f:
                        for obj in qas:
                            f.write(json.dumps(obj) + '\n')
            i += 1
    return True

def main():
    read_papers_from_csv()

if __name__ == "__main__":
    main()