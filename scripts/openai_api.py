import os
import openai
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Define the OpenAI API parametersx
# openai.api_key = '<YOUR_OPENAI_API_KEY>'
openai.api_key = os.environ.get('OPENAI_API_KEY')
BACKUP_KEY = os.environ.get('OPENAI_API_KEY2')

NUM_KEYS = 3
KEYS = []
CURR_KEY = 0

for i in range(NUM_KEYS):
    if i == 0:
        KEYS.append(os.environ.get('OPENAI_API_KEY'))
    else:
        KEYS.append(os.environ.get(f'OPENAI_API_KEY{i+1}'))

# print(len(KEYS))

def get_chunk_tokens(text, MAX_CHUNK=3000):
    cur_len = token_count(text)
    if cur_len > MAX_CHUNK:
        char_length = (len(text)*MAX_CHUNK)//cur_len
        text = text[:char_length]
    return text

def token_count(text:str, model='gpt-3.5-turbo'):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model="text-embedding-ada-002")['data'][0]['embedding']
    
def chatGPT(chat_history, system='You are a helpful assistant.', engine='gpt-3.5-turbo'):
    global CURR_KEY
    messages = [
        {"role": "system", "content": system},
    ]
    
    messages = messages + [{"role":"user", "content":chat_history}]
    
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