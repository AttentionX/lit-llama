import openai_api

def generate_qa_pairs(paper_info:list, paper_text:str, model='gpt-3.5-turbo-16k'):
    context = f'The following is from the paper "{paper_info[0]}" released in {paper_info[1]}'
    examples = [
        '{"question": "What is the Recurrent Memory Transformer?", "answer": "The Recurrent Memory Transformer is a model architecture that retains information across up to 2 million tokens by augmenting a pre-trained BERT model with recurrent memory, allowing for the storage and processing of both local and global information and enabling information flow between segments of the input sequence through the use of recurrence."}',
        '{"question": "What is the SAM dataset and how many images and masks was it trained on?", "answer": "The SAM dataset is a large-scale dataset used to train the Segment Anything Model (SAM). It was trained on 11 million images and 1.1 billion masks."}'
        '{"question": "What is the purpose of XMem in TAM?", "answer": "XMem is used for long-term video object segmentation with an Atkinson-Shiffrin memory model to refine subsequent object discrimination."}',
        '{"question": "How does LIMA\'s performance compare to GPT-4?", "answer": "In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases. However, humans typically prefer responses from GPT-4, Claude, and Bard over LIMA."}',
        '{"question": "How was LIMA improved?", "answer": "LIMA was improved by gathering 30 multi-turn dialogue chains that were fine-tuned on a new version of LIMA from the pretrained LLaMA model using the combined 1,030 examples."}'
    ]
    examples = '\n'.join(examples)
    instruction = f'Generate as many question-answer pairs as possible, in the following format:\n---\n{examples}\n---\n about the following information from the paper, covering all of the details mentioned, including all information related to deep learning, natural language processing, or models. Output in json format (Each line containing a json object).'
    rules = f'''Make sure to follow the following rules:
    1. Phrase questions such that they can be answered independently, without context. (ex. do not ask questions like "What is figure 4?" or "Who is the author?" or "What is the main contribution of this work?" or "What is the main contribution of the paper?" because those questions cannot be answered independently without context. Ther person answering will not know what you are referring to by "work" or "paper" when those kinds of questions are asked on its own. Be sure to either provide sufficient context or rephrase the question that can be answered independently). 
    2. Each q-a pairs should be in json format in one line, each separated by a newline as shown in the example above
    3. Be comprehensive! Make sure to cover all of the essential information in the passage and every related concepts and information and generate as many question answer pairs as possible covering every detail and component mentioned in the paper
    4. Make sure to cover every referenced model, dataset, or method mentioned in the paper
    5. Think step by step and be creative in asking questions (one question could have multiple queries, it can be a short response question- doesn't have to start with 'what' or 'how', multiple choice, fill in the blank, some questions could require more reasoning and inference rather than just regurgitating what is stated and many other methods to question whether the reader knows the full extent of the paper)
    '''
    prompt = f'{context}\n\n{instruction}\n\n{rules}\n\n{paper_text}'
    answer = openai_api.chatGPT(prompt, engine=model)
    return answer