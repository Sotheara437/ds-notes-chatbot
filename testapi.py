import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

print('Step 1: Script started')

api_key = os.getenv('GROQ_API_KEY')
print('Step 2: Key loaded:', bool(api_key))

client = Groq(api_key=api_key)
print('Step 3: Client created')

response = client.chat.completions.create(
    model='llama-3.1-8b-instant',
    messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}]
)

print('Step 4: Response received')
print('? Groq API is working!')
print('Response:', response.choices[0].message.content)
