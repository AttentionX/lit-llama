import sys
from pathlib import Path
import asyncio

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
from custom_parallel_request import process_requests
from qa_prompts import generate_qa_pairs

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

# def generate_qa_pairs(paper_info:list, paper_text:str, model='gpt-3.5-turbo-16k'):
#     context = f'The following is from the paper "{paper_info[0]}" released in {paper_info[1]}'
#     examples = [
#         '{"question": "What is the Recurrent Memory Transformer?", "answer": "The Recurrent Memory Transformer is a model architecture that retains information across up to 2 million tokens by augmenting a pre-trained BERT model with recurrent memory, allowing for the storage and processing of both local and global information and enabling information flow between segments of the input sequence through the use of recurrence."}',
#         '{"question": "What is the SAM dataset and how many images and masks was it trained on?", "answer": "The SAM dataset is a large-scale dataset used to train the Segment Anything Model (SAM). It was trained on 11 million images and 1.1 billion masks."}'
#         '{"question": "What is the purpose of XMem in TAM?", "answer": "XMem is used for long-term video object segmentation with an Atkinson-Shiffrin memory model to refine subsequent object discrimination."}',
#         '{"question": "How does LIMA\'s performance compare to GPT-4?", "answer": "In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases. However, humans typically prefer responses from GPT-4, Claude, and Bard over LIMA."}',
#         '{"question": "How was LIMA improved?", "answer": "LIMA was improved by gathering 30 multi-turn dialogue chains that were fine-tuned on a new version of LIMA from the pretrained LLaMA model using the combined 1,030 examples."}'
#     ]
#     examples = '\n'.join(examples)
#     instruction = f'Generate as many question-answer pairs as possible, in the following format:\n---\n{examples}\n---\n about the following information from the paper, covering all of the details mentioned, including all information related to deep learning, natural language processing, or models. Output in json format (Each line containing a json object).'
#     rules = f'''Make sure to follow the following rules:
#     1. Phrase questions such that they can be answered independently, without context. (ex. do not ask questions like "What is figure 4?" or "Who is the author?" or "What is the main contribution of this work?" or "What is the main contribution of the paper?" because those questions cannot be answered independently without context. Ther person answering will not know what you are referring to by "work" or "paper" when those kinds of questions are asked on its own. Be sure to either provide sufficient context or rephrase the question that can be answered independently). 
#     2. Each q-a pairs should be in json format in one line, each separated by a newline as shown in the example above
#     3. Be comprehensive! Make sure to cover all of the essential information in the passage and every related concepts and information and generate as many question answer pairs as possible covering every detail and component mentioned in the paper
#     '''
#     prompt = f'{context}\n\n{instruction}\n\n{rules}\n\n{paper_text}'
#     answer = openai_api.chatGPT(prompt, engine=model)
#     return answer

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

def get_qa_for_paper(paper_json:dict, save_output=True):
    url = paper_json['url']
    title = paper_json['title']
    date = paper_json['date']
    destination_path = paper_json['save_filepath']
    out_title = paper_json['out_title']
    model = paper_json['model']

    pages = prepare_arxiv.retrieve_pdf_from_url(url)

    paper_text = '\n\n'.join(pages)
    paper_text = openai_api.get_chunk_tokens(paper_text, max_tokens=12000, model=model)

    qas = generate_qa_pairs([title, date], paper_text)
    qas = qas.split('\n')
    qas = [json.loads(qa) for qa in qas if '}' in qa]

    if save_output:
        mode = 'w'
        if os.path.exists(destination_path / f'{out_title}.jsonl'):
            mode = 'a'
        with open(destination_path / f'{out_title}.jsonl', mode) as f:
            for obj in qas:
                f.write(json.dumps(obj) + '\n')

def process_papers_parallel(file_path:str='data/notable/adam.txt', model='gpt-3.5-turbo-16k'):
    tasks = []

    destination_path = Path('data/qa/notable')
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            # print(i+1, 'th file in', file_path.split('/')[-1])
            if len(line) >= 4:
                url = line[3].strip()
                date = line[1].strip()
                title = line[0].strip()

                if 'arxiv' in url and 'abs' in url:
                    url = url.replace('abs', 'pdf')

                if not url.startswith('http'):
                    if title != 'title':
                        print(f'Line {i+1} in {file_path.split("/")[-1]} has invalid link: {line}')
                    i += 1
                    continue
                
                clean_title = title.replace(' ', '_').replace(':','_').replace('!','')
                out_title = f'{i}_{clean_title}.jsonl'

                save_filepath = destination_path / out_title
                save_filepath = str(save_filepath)

                paper_json = {
                    'title': title,
                    'date': date,
                    'url': url,
                    'save_filepath': save_filepath,
                    'out_title': out_title,
                    'model': model,
                }
                tasks.append(paper_json)
            else:
                print(f'Line {i+1} in {file_path.split("/")[-1]} is invalid: {line}')
            i += 1
    asyncio.run(
        process_requests(
            tasks=tasks,
            engine='gpt-3.5-turbo-16k'
        )
    )

def read_papers_from_csv(csv_path:str='data/notable/adam.txt', all_at_once=True, model='gpt-3.5-turbo-16k'):
    destination_path = Path('data/qa/notable')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            print(i+1, 'th file in', csv_path.split('/')[-1])
            if len(line) >= 4:
                url = line[3].strip()
                date = line[1].strip()
                title = line[0].strip()

                if not url.startswith('http'):
                    i += 1
                    print(f'Line {i+1} in {csv_path.split("/")[-1]} has invalid link: {line}')
                    continue
                
                clean_title = title.replace(' ', '_')
                out_title = f'{i+1}_{clean_title}.jsonl'

                save_filepath = destination_path / out_title
                save_filepath = str(save_filepath)

                paper_json = {
                    'title': title,
                    'date': date,
                    'url': url,
                    'save_filepath': save_filepath,
                    'out_title': out_title,
                    'model': model,
                }
                
                if all_at_once:
                    get_qa_for_paper(paper_json)
                    
                    # pages = prepare_arxiv.retrieve_pdf_from_url(url)

                    # paper_text = '\n\n'.join(pages)
                    # paper_text = openai_api.get_chunk_tokens(paper_text, max_tokens=12000, model=model)

                    # qas = generate_qa_pairs([title, date], paper_text)
                    # qas = qas.split('\n')
                    # qas = [json.loads(qa) for qa in qas if '}' in qa]
                    # mode = 'w'
                    # if os.path.exists(destination_path / f'{out_title}.jsonl'):
                    #     mode = 'a'
                    # with open(destination_path / f'{out_title}.jsonl', mode) as f:
                    #     for obj in qas:
                    #         f.write(json.dumps(obj) + '\n')
                else:
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
            else:
                print(f'Line {i+1} in {csv_path.split("/")[-1]} is invalid: {line}')
            i += 1
    return True

def main():
    process_papers_parallel()
    # read_papers_from_csv()

if __name__ == "__main__":
    main()