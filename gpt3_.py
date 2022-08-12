import os
import sys
from termcolor import colored
import openai
import click

#You need a OpenAI api key: https://beta.openai.com/account/api-keys   Sign up, free $18


click.command()
def text_generation():
    prompt = click.prompt('Specify prompt', default='')
    print('\n\n')
    instruction = click.prompt('Specify instruction', default='')
    api = os.getenv('openapi')   # I hided my api key in here
    openai.api_key = api         # A quik fix is to paste the API here and delete the above line
    """
    Function: text_generation
    -----------------------
    This function takes in a text and returns the generated text.
    param text: The text to be generated.
    type text: str
    return: generated_text: The generated text.
    type generated_text: str
    """

    prompt = '# '+prompt.lstrip('\n') + '\n# '+instruction + '\n\"\"\"'

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt= prompt,
    temperature=0.9,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=2,
    presence_penalty=2
    )
    os.system('clear')

    text=colored(response['choices'][0]['text'], 'red', attrs=['reverse', 'blink'])
    print(text)


if __name__ == '__main__':
    text_generation()
    
