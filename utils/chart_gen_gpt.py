import os
import csv
import ast
import httpx
import json
import time
from openai import OpenAI
from tqdm import tqdm

def list_filenames_without_extension(directory):
    filenames = []
    for filename in os.listdir(directory):
        # Get the file name without the extension
        name, _ = os.path.splitext(filename)
        filenames.append(name)
    return filenames

def load_csv_as_text(csv_file_path):
    text_data = []
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert the row list to a comma-separated string
            text_data.append(','.join(row))
    # Join all rows with newline characters to create a single text block
    return '\n'.join(text_data)

def update_base_url(request: httpx.Request) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path="/v1/chat/gpt4-8k") # chat/gpt4-8k /chat

os.environ['AALTO_OPENAI_API_KEY'] = ''
assert (
    "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
client = OpenAI(
    base_url="https://aalto-openai-apigw.azure-api.net",
    api_key=False, # API key not used, and rather set below
    default_headers = {
        "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
    },
    http_client=httpx.Client(
        event_hooks={ "request": [update_base_url] }
    ),
)


table_directory_path = './chartqa/tables'
json_directory_path = './chartqa/annotations'
vegalite_directory_path = './chartqa/vega'


if __name__ == '__main__':
    filenames_without_extension = list_filenames_without_extension(table_directory_path)
    for filename in tqdm(filenames_without_extension):
        query_file = open(f'{json_directory_path}/{filename}.json', 'r')
        json_data = json.load(query_file)
        question = json_data['tasks'][0]['question']

        csv_file_path = f'{table_directory_path}/{filename}.csv'
        csv_text = load_csv_as_text(csv_file_path)

        message = [
            {
                "role": "system", 
                "content": "Generate a bar chart given this data table, facilitate people to complete the task of \"%s\" \
                            Data table: %s \
                            You should generate the bar chart as a vega-lite json format. \
                            The json should specify the height to be 600px, and a calculated width for ascetics."%(question, csv_text)
            },
            {
                "role": "user", 
                "content": "Generate a bar chart given this data table, facilitate people to complete the task of \"%s\" \
                            You should generate the bar chart as a vega-lite json format. \
                            The json should specify the height to be 600px, and a calculated width for ascetics. \
                            Please only provide a valid json format without any additional text!"%(question)
            }
        ]
        max_num_requests = 5
        for attempt in range(max_num_requests):
            completion = client.chat.completions.create(
                model="no_effect", # the model variable must be set, but has no effect, model selection done with URL
                messages=message,
            )
            message_content = completion.choices[0].message.content
            try:
                output = json.loads(message_content)
                out_file = open("%s/%s.json"%(vegalite_directory_path, filename), "w")
                json.dump(output, out_file)
                break
            except:
                message[-1]["content"] += "Please only provide a valid json format without any additional text!"
                time.sleep(10)
        time.sleep(10)
