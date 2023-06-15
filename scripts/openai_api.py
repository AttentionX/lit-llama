import os
import openai
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Define the OpenAI API parametersx
# openai.api_key = '<YOUR_OPENAI_API_KEY>'
openai.api_key = os.environ.get('OPENAI_API_KEY')
BACKUP_KEY = os.environ.get('OPENAI_API_KEY2')

NUM_KEYS = 1
KEYS = []
CURR_KEY = 0

for i in range(NUM_KEYS):
    if i == 0:
        KEYS.append(os.environ.get('OPENAI_API_KEY'))
    else:
        KEYS.append(os.environ.get(f'OPENAI_API_KEY{i+1}'))

# print(len(KEYS))

def get_chunk_tokens(text, max_tokens=3000, model='gpt-3.5-turbo'):
    cur_len = token_count(text, model=model)
    if cur_len > max_tokens:
        char_length = (len(text)*max_tokens)//cur_len
        text = text[:char_length]
    return text

def token_count(text:str, model='gpt-3.5-turbo'):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model="text-embedding-ada-002")['data'][0]['embedding']
    
def chatGPT(chat_history, system='You are a helpful assistant.', engine='gpt-3.5-turbo', switch_keys=False):
    global CURR_KEY
    messages = [
        {"role": "system", "content": system},
    ]
    
    messages = messages + [{"role":"user", "content":chat_history}]

    if not switch_keys:
        response = openai.ChatCompletion.create(
            model = engine,
            messages = messages,
        )

        answer = response.choices[0].message.content
        return answer
    
    i = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model = engine,
                messages = messages,
            )
        except Exception as e:
            NEW_KEY = (CURR_KEY + 1) % NUM_KEYS
            print(f'Switching to api key {NEW_KEY}')
            openai.api_key = KEYS[NEW_KEY]
            CURR_KEY = NEW_KEY
            # if openai.api_key == BACKUP_KEY:
            #     print('OpenAI API Switching to Original Key')
            #     openai.api_key = os.environ.get('OPENAI_API_KEY')
            # else:
            #     print('OpenAI API Switching to Backup Key')
            #     openai.api_key = BACKUP_KEY
            i += 1
            if i == NUM_KEYS:
                raise Exception(e)
        else:
            break
    
    answer = response.choices[0].message.content
    return answer