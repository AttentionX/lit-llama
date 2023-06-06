"""Prepare qa datasets from conference papers"""

from pathlib import Path
from typing import List
import requests
import io
import PyPDF2
import os
import csv
import re
from datetime import datetime
import json
from scripts import openai_api
import math

# from scripts.prepare_arxiv import prepareQA

CONFERENCE_PAPER_METADATA_PATH = Path("data/conference")
QA_DATA_PATH = Path("data/qa")

MAX_PAGES = 30


class Paper:
    """Paper class"""

    def __init__(
        self, title: str, published: str, modified: str, url: str, conference: str
    ):
        self.title: str = title
        self.published: str = published
        self.modified: str = modified
        self.url: str = url
        self.conference: str = conference


def fetch_conference_papers(path: str) -> List[Paper]:
    """Fetch conference papers from csv file"""
    papers: List[Paper] = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            if i == 0:
                i += 1
                continue
            print(i + 1, "th file in", path.split("/")[-1])
            if len(line) >= 3:
                paper = Paper(
                    title=line[0],
                    published=line[1],
                    modified=line[2],
                    url=line[3],
                    conference=line[4],
                )
                papers.append(paper)
            i += 1
    return papers


def prepare_qa_for_paper(paper: Paper) -> None:
    """Retrieve paper from url"""
    # Download the PDF file from the URL
    response = requests.get(paper.url)

    # Extract the content of the PDF file as bytes
    content = io.BytesIO(response.content)

    # Read the PDF file using PyPDF2
    pdf_reader = PyPDF2.PdfReader(content)

    # Extract the text from the PDF file
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
        if page_num == MAX_PAGES:
            break

    date_str = paper.published.replace(",", "")
    try:
        published_date = datetime.strptime(date_str, "%d %B %Y")
    except ValueError:
        published_date = datetime.strptime(date_str, "%d %b %Y")
    formatted_date = datetime.strftime(published_date, "%y%m")

    file_title = formatted_date + "_" + paper.title
    print(f"Saving {paper.title}")
    text_path = f"data/raw_text/{file_title}.txt"
    if not os.path.exists(text_path):
        with open(text_path, "w") as file:
            file.write(text)

    print(f"Saved {paper.title}")
    prepareQA(
        text, Path(QA_DATA_PATH, Path(paper.conference)), file_title, formatted_date
    )


# Prepare dataset in q&a format
def prepareQA(
    text, destination_path: Path, title, date=None, chunk_size=7000, skip_if_exists=True
):
    if os.path.exists(destination_path / f"{title}.jsonl") and skip_if_exists:
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
        print(
            f"Processing chunk {i+1} in {title} dataset of length {math.ceil(len(text)/chunk_size)}"
        )
        end_index = (
            (i + 1) * chunk_size if (i + 1) * chunk_size < len(text) else len(text)
        )
        chunk = text[i * chunk_size : end_index]

        full_title = text.split("\n")[0]
        context = (
            f"The given information is an excerpt from the paper titled {full_title}"
        )
        if date is not None:
            year = f"20{date[:2]}"
            month = date[2:4]
            context += f" published in {month} {year}.\n"

        instruction = f"{context}\n{instruction}"

        prompt = f"{instruction}\n\nExample Q&A:\n{qa_examples}\n\nInformation:\n{chunk}\n\nQ&A:"
        answer = openai_api.chatGPT(prompt)
        # print(answer)
        answer = answer.split("\n")
        jsonl.extend(answer)
        i += 1
        # if i == 5:
        #     break

    print(f"Saving {title} dataset of qa pairs {len(jsonl)}")
    # Save answer to jsonl file, switch to append mode 'a' if file already exists
    mode = "w"
    # Create conference folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    if os.path.exists(destination_path / f"{title}.jsonl"):
        mode = "a"
    with open(destination_path / f"{title}.jsonl", mode) as f:
        for obj in jsonl:
            if obj[-1] != "}":
                continue
            data = json.loads(obj)
            f.write(json.dumps(data) + "\n")


def main() -> None:
    """Main function"""
    for file_path in CONFERENCE_PAPER_METADATA_PATH.iterdir():
        if file_path.suffix == ".txt":
            papers = fetch_conference_papers(file_path.as_posix())
            for paper in papers:
                prepare_qa_for_paper(paper)
            # prepare_qa_for_paper(papers[0])


if __name__ == "__main__":
    main()
