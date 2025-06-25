from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import openai
import json
import logging
import tiktoken
import time

load_dotenv(".env")

class GPTmethods:
    def __init__(self, model_id='gpt-4.1-2025-04-14'):
        openai.api_key = os.environ.get("OPENAI_API_KEY")  # Access environment variable
        self.model_id = model_id
        self.pre_path = 'data/processed/'

    # Create a training and validation JSONL file for GPT fine-tuning
    def create_jsonl(self, data_type, data_set):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + data_set)
        data = [] 

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            data.append(
                {
                    "messages": [
                        {"role": "system",
                         "content": "You are a spam filter."},
                        {"role": "user",
                         "content": 'Please parse the text and classify it. Return your response in JSON format as either spam {"Spam":1} or non-spam {"Spam":0}. Text:\n' +
                                    row['text'] + ''}, 
                        {"role": "assistant", "content": '{"Spam":' + str(row['label']) + '}'} 
                    ]
                } 
            )

        output_file_path = self.pre_path + "ft_dataset_gpt_" + data_type + ".jsonl"  # Define the path
        
        # Write data to the JSONL file
        with open(output_file_path, 'w') as output_file:
            for record in data:
                
                # Convert the dictionary to a JSON string and write it to the file
                json_record = json.dumps(record)
                output_file.write(json_record + '\n')

        return {"status": True, "data": f"JSONL file '{output_file_path}' has been created."}

    # Create a conversation with GPT model
    def gpt_conversation(self, conversation):
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model_id,
            messages=conversation,
            seed=420,
        )
        return completion.choices[0].message

    # Clean the response
    def clean_response(self, response, a_field):

        # Search for JSON in the response
        start_index = response.find('{')
        end_index = response.rfind('}')

        if start_index != -1 and end_index != -1:
            json_str = response[start_index:end_index + 1]
            try:
                # Attempt to load the extracted JSON string
                json_data = json.loads(json_str)
                return {"status": True, "data": json_data}
            except json.JSONDecodeError as e:
                # If an error occurs during JSON parsing, handle it
                logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                              f"resulted in the following response: {response}")
                return {"status": False,
                        "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                                f"resulted in the following response: {response}"}
        else:
            logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
                          f"following response: {response}")
            return {"status": False, "data": f"No JSON found in the response. The input '{a_field}', "
                                             f"resulted in the following response: {response}"}

    # Prompt the GPT model to make a prediction
    def gpt_prediction(self, input):
        conversation = []
        conversation.append({'role': 'system',
                             'content': "You are a spam filter."}) 
        conversation.append({'role': 'user',
                             'content': 'Please parse the text and classify it. Return your response in JSON format as either spam {"Spam":1} or non-spam {"Spam":0}. Text:\n' +
                                        input['text'] + ''})  
        conversation = self.gpt_conversation(conversation) 
        content = conversation.content

        # Clean the response and return
        return self.clean_response(response=content, a_field=input['text'])  

    # Make predictions for a specific data_set appending a new prediction_column
    def predictions(self, data_set, prediction_column):

        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + data_set)

        file_name_without_extension = os.path.splitext(os.path.basename(data_set))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original.csv'
        if not os.path.exists(original_file_path):
            os.rename(self.pre_path + data_set, original_file_path)

        # Check if the prediction_column is already present in the header
        if prediction_column not in df.columns:  
            df[prediction_column] = pd.NA
            df = df.astype({prediction_column: 'Int64'})

        # Update the CSV file with the new header (if columns were added)
        if prediction_column not in df.columns:
            df.to_csv(self.pre_path + data_set, index=False)

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            if pd.isnull(row[prediction_column]):
                prediction = self.gpt_prediction(input=row)
                if not prediction['status']:
                    print(prediction)
                    break
                else:
                    print(prediction)

                    if prediction['data']['Spam'] != '':  
                        df.at[index, prediction_column] = int(prediction['data']['Spam'])  
                        df.to_csv(self.pre_path + data_set, index=False)
                    else:
                        logging.error(
                            f"No rating instance was found within the data for '{row['text']}', and the " 
                            f"corresponding prediction response was: {prediction}.") 
                        return {"status": False,
                                "data": f"No rating instance was found within the data for '{row['text']}', " 
                                        f"and the corresponding prediction response was: {prediction}."}  
        df[prediction_column] = df[prediction_column].astype('Int64') 

        return {"status": True, "data": 'Prediction have successfully been'}

    # Upload Dataset for GPT Fine-tuning
    def upload_file(self, dataset):
        upload_file = openai.File.create(
            file=open(dataset, "rb"),
            purpose='fine-tune'
        )
        return upload_file

    # Train GPT model
    def train_gpt(self, file_id):
        return openai.FineTuningJob.create(training_file=file_id, model="gpt-4.1-2025-04-14")

    # Delete Fine-Tuned GPT model
    def delete_finetuned_model(self, model): 
        return openai.Model.delete(model)

    # Cancel Fine-Tuning
    def cancel_gpt_finetuning(self, train_id):  
        return openai.FineTuningJob.cancel(train_id)

    # Get all Fine-Tuned models and their status
    def get_all_finetuned_models(self):
        return openai.FineTuningJob.list(limit=10)

# Configure logging to write to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

# Instantiate the GPTmethods class
GPT = GPTmethods()
GPT.create_jsonl(data_type='train', data_set='train.csv') 
GPT.create_jsonl(data_type='test', data_set='test.csv')
GPT.create_jsonl(data_type='val', data_set='val.csv')

# Make predictions before Fine-tuning using the Base Model
dataset_test = "data/processed/ft_dataset_gpt_test.jsonl"
tokenizer = tiktoken.encoding_for_model("gpt-4") 
token_counts = []

df = pd.read_csv("data/processed/test.csv")

model_ids = [
    "gpt-4.1-mini-2025-04-14",
    # "ft:gpt-4.1-mini-2025-04-14:sunny:spam:BV5bIsAw", 
    "gpt-4.1-nano-2025-04-14",
    # "ft:gpt-4.1-nano-2025-04-14:sunny:spam:BVCRzCEZ", 
    "gpt-4o-mini-2024-07-18",
    # "ft:gpt-4o-mini-2024-07-18:sunny:spam:BV62vAVr", 
]

# Metrics
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, roc_curve
)
import numpy as np

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)

SCORING = {
    "F1":             f1_score,
    "Accuracy":       accuracy_score,
    "Precision":      precision_score,
    "Recall":         recall_score,
    "MCC":            matthews_corrcoef,
    "ROC AUC":        roc_auc_score,
    "PRC AREA":       average_precision_score,
    "FPR":            false_positive_rate,
}

columns = list(SCORING.keys()) + ["inference_time"]
scores = pd.DataFrame(columns=columns)

# Compute metrics for each model
for model_id in model_ids:
    print(f"Evaluating {model_id}…")
    gpt = GPTmethods(model_id=model_id)
    start = time.time()
    preds = []
    for text in df["text"]:
        out = gpt.gpt_prediction({"text": text})
        preds.append(int(out["data"]["Spam"]))
    inference_time = time.time() - start
    tmp = df.copy()
    tmp[model_id] = preds
    row = {}
    for name, fn in SCORING.items():
        try:
            row[name] = fn(tmp["label"], tmp[model_id])
        except Exception as e:
            print(f"Warning: Could not compute {name} for {model_id}: {e}")
            row[name] = None  
    row["inference_time"] = round(inference_time, 4)

    scores.loc[model_id] = [
        round(row[c], 4) if isinstance(row[c], float) else row[c]
        for c in columns
    ]
print(scores)