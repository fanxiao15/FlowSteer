import base64
import requests
from openai import OpenAI
import sys


OPENAI_KEY = ""
client = OpenAI(api_key=OPENAI_KEY, base_url='')

def language_reasoning(prompt):
   
    completion = client.chat.completions.create(

                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1
            )
    response = completion.choices[0].message.content
    return response

