import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Define the OpenAI API parametersx
# openai.api_key = '<YOUR_OPENAI_API_KEY>'
openai.api_key = os.environ.get('OPENAI_API_KEY')
BACKUP_KEY = os.environ.get('OPENAI_API_KEY2')


def embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model="text-embedding-ada-002")['data'][0]['embedding']
    
def chatGPT(chat_history, system='You are a helpful assistant.'):
    engine = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": system},
    ]
    
    messages = messages + [{"role":"user", "content":chat_history}]
    
    response = ''
    i = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model = engine,
                messages = messages,
            )
        except Exception as e:
            print(e)
            if openai.api_key == BACKUP_KEY:
                openai.api_key = os.environ.get('OPENAI_API_KEY')
            else:
                openai.api_key = BACKUP_KEY
            i += 1
            if i == 2:
                raise Exception(e)
        else:
            break
    
    answer = response.choices[0].message.content
    return answer