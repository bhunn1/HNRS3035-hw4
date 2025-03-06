import os
import chromadb
import json
import time
import re
from openai import OpenAI
from openai import AzureOpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(".env")

# Load dataset
with open('dev-v2.0 (1).json', 'r') as f:
    data = json.load(f)

q_and_as = {}
for topic in data["data"]:
    for paragraph in topic["paragraphs"]:
        for qas in paragraph["qas"]:
            if not qas["is_impossible"]:
                question = qas["question"]
                q_id = qas["id"]
                ans = []
                for answer in qas["answers"]:
                    ans.append(answer["text"])
                q_and_as[(question, q_id)] = ans
            if len(q_and_as) >= 500:
                break
        if len(q_and_as) >= 500:
            break
    if len(q_and_as) >= 500:
        break

def get_mini_responses():
    # Create Chroma client
    chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    print('Embedding function initialized.')

    collection = chroma_client.get_collection(
        name="contexts",
        embedding_function=openai_ef
    )
    print('Chroma collection created.')


    context = []

    for topic in data["data"]:
        for paragraph in topic["paragraphs"]:
            context.append(paragraph["context"])

    ids = list(str(i) for i in range(len(context)))
    print('Contexts and Q&As loaded.')

    """
    collection.add(
        ids=ids,
        documents=context,
    )"""

    system_prompt = """You are a genius AI assistant that answers questions concisely using data returned by a search engine.

    Guidelines:
    \t1. You will be provided with a question by the user, you must answer that question, and nothing else.
    \t2. Your answer should come directly from the provided context from the search engine.
    \t3. Do not make up any information not provided in the context.
    \t4. If the provided question does not contain the answers, respond with 'I am sorry, but I am unable to answer that question.'
    \t5. Be aware that some chunks in the context may be irrelevant, incomplete, and/or poorly formatted.

    Here is the provided context:
    {context}

    Here is the question: {question}

    Your response: """

    count = 0
    tasks = []
    for (q, q_id), a in q_and_as.items():

        print(f'Question: {q}, Answers: {a}')
        # Query the collection for the top 5 contexts related to the prompt question
        result = collection.query(query_texts=q, n_results=5)
        print(f'Found top 5 contexts for Question: {q}')

        messages = [
            {"role": "system", "content": system_prompt.format(context=result['documents'], question=q)},
        ]

        custom_id = f"question={q_id}"

        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 100,
                "messages": messages
            }
        }
        tasks.append(task)


    # Here, we are writing a local file to store the tasks. This is a jsonl file, newline delimited)
    with open("C:\\CSC4700\\Homework4\\gpt-4o-mini-input.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    # establish OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to openai
    batch_file = client.files.create(
        file=open("C:\\CSC4700\\Homework4\\gpt-4o-mini-input.jsonl", 'rb'),
        purpose='batch'
    )

    # Run the batch using the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # loop until the status of our batch is completed
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)
    print("Done processing batch.")

    print("Writing data...")
    # Write the results to a local file in jsonl format
    result = client.files.content(check.output_file_id).content
    output_file_name = "C:\\CSC4700\\Homework4\\gpt-4o-mini-RAG-feb18.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    # load the output file, extract each sample output, and append to a list
    results = []
    with open(output_file_name, 'r') as file:
        for line in file:
            # this converts the string into a Json object
            json_object = json.loads(line.strip())
            results.append(json_object)

    # Show the responses
    for item in results:
        print("Model's Response:")
        print('\t', item['response']['body']['choices'][0]['message']['content'])


#======================================================================================================================

def get_llama_responses():

    chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    print('Embedding function initialized.')

    collection = chroma_client.get_collection(
        name="contexts",
        embedding_function=openai_ef
    )
    print('Chroma collection created.')

    context = []

    for topic in data["data"]:
        for paragraph in topic["paragraphs"]:
            context.append(paragraph["context"])

    ids = list(str(i) for i in range(len(context)))
    print('Contexts and Q&As loaded.')

    """
    collection.add(
        ids=ids,
        documents=context,
    )"""

    # Establish client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-06-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


    system_prompt = """You are a genius AI assistant that answers questions concisely using data returned by a search engine.

        Guidelines:
        \t1. You will be provided with a question by the user, you must answer that question, and nothing else.
        \t2. Your answer should come directly from the provided context from the search engine.
        \t3. Do not make up any information not provided in the context.
        \t4. If the provided question does not contain the answers, respond with 'I am sorry, but I am unable to answer that question.'
        \t5. Be aware that some chunks in the context may be irrelevant, incomplete, and/or poorly formatted.

        Here is the provided context:
        {context}

        Here is the question: {question}

        Your response: """

    answers = []
    for (q, q_id), a in q_and_as.items():

        # Query the collection for the top 5 contexts related to the prompt question
        result = collection.query(query_texts=q, n_results=5)
        print(f'Found top 5 contexts for Question: {q}')

        # Use Llama to create an answer with the given prompts
        response = client.chat.completions.create(
                model="Llama-3.2-11B-Vision-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt.format(context=result['documents'], question=q)},
                ],
                temperature=0.7,
                max_tokens=100
        )

        print(response)
        answers.append(response.choices[0].message.content)

    # Save the answers into a file
    output_file_name = "C:\\CSC4700\\Homework4\\llama-RAG-feb18.jsonl"
    with open(output_file_name, 'w') as file:
        for i in answers:
            file.write(json.dumps(i) + '\n')
    print(f'Responses saved to output file.')

    results = []
    with open(output_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)

    print(f'Model answers: {answers}')


def score_mini():
    """
        method that scores the accuracy of the GPT-4o-mini answers

        :return:
        """

    mini_answers = []
    with open('C:\\CSC4700\\Homework4\\gpt-4o-mini-RAG-feb18.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            mini_results = data["response"]["body"]["choices"][0]["message"]["content"].strip()
            mini_answers.append(mini_results)

    scoring_prompt = (
        "You are a teacher tasked with determining whether a student’s answer to a question was correct, based "
        "on a set of possible correct answers. You must only use the provided correct answers to determine if "
        "the student's response was correct.")
    user_prompt = ("Question: {question}\n\n"
                   "Student’s Response: {student_response}\n\n"
                   "Possible Correct Answers:{correct_answers}\n\n"
                   "Your response should only be a valid Json as shown below:\n"
                   "{{\n"
                   "    \"explanation\" (str): \"A short explanation of why the student’s answer was correct or incorrect.\",\n"
                   "    \"score\" (bool): \"true if the student’s answer was correct, false if it was incorrect.\"\n"
                   "}}\n\n"
                   "Your response:")

    # Loop through the Q&A's and form the API calls to gpt-4o with the proper prompts and parameters
    tasks = []
    count = 0
    for (q, q_id), a in q_and_as.items():

        messages = [
            {"role": "system", "content": scoring_prompt},
            {"role": "user",
             "content": user_prompt.format(question=q, student_response=mini_answers[count], correct_answers=a)}
        ]

        custom_id = q_id
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 100,
                "response_format": {"type": "json_schema",
                                    "json_schema": {
                                        "name": "score_response",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "explanation": {
                                                    "type": "string",
                                                },
                                                "score": {
                                                    "type": "boolean",
                                                }
                                            },
                                            "required": ["explanation", "score"],
                                            "additionalProperties": False
                                        },
                                    }
                                    },
                "messages": messages
            }
        }
        tasks.append(task)
        count += 1


    # Dump the llama model's answers into a jsonl file for the batch
    with open("../mini_score.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')
    print(f'Batch input file created.')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to openai
    batch_file = client.files.create(
        file=open("../mini_score.jsonl", 'rb'),
        purpose='batch'
    )
    batch_id = batch_file.id
    print(f'Batch file uploaded successfully. File ID: {batch_id}')

    # Submit the batch job to the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_job_id = batch_job.id
    print(f'Batch job submitted. Batch job ID: {batch_job_id}')

    # Check status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job_id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        if check.status == 'failed':
            print(f'Batch failed! {check}')
            exit()
        time.sleep(30)
    print("Done processing batch.")

    # Get the results back from GPT-4o
    results = []
    result_bytes = client.files.content(check.output_file_id).content
    result_str = result_bytes.decode('utf-8')
    print(result_str)  # Debugging: Check the raw response

    for line in result_str.splitlines():
        data = json.loads(line)  # Parse each line as JSON
        content_str = data["response"]["body"]["choices"][0]["message"]["content"].strip()

        # Parse the content as JSON
        content_json = json.loads(content_str)

        # Extract the score
        score = content_json.get("score", False)  # Default to False if missing
        results.append(score)

    print(results)
    # Count correct answers
    count_correct = sum(1 for score in results if score)

    print(f"GPT-Mini's accuracy was {(count_correct / len(results)) * 100}")
    return ((count_correct / len(results)) * 100)


def score_llama():
    """
    method that scores the accuracy of the Llama 3.2 11B Vision Instruct answers

    :return:
    """

    llama_answers = []
    with open('C:\\CSC4700\\Homework4\\llama-RAG-feb18.jsonl', 'r') as f:
        for line in f:
            llama_answers.append(json.loads(line.strip()))

    scoring_prompt = (
        "You are a teacher tasked with determining whether a student’s answer to a question was correct, based "
        "on a set of possible correct answers. You must only use the provided correct answers to determine if "
        "the student's response was correct.")
    user_prompt = ("Question: {question}\n\n"
                   "Student’s Response: {student_response}\n\n"
                   "Possible Correct Answers:{correct_answers}\n\n"
                   "Your response should only be a valid Json as shown below:\n"
                   "{{\n"
                   "    \"explanation\" (str): \"A short explanation of why the student’s answer was correct or incorrect.\",\n"
                   "    \"score\" (bool): \"true if the student’s answer was correct, false if it was incorrect.\"\n"
                   "}}\n\n"
                   "Your response:")

    # Loop through the Q&A's and form the API calls to gpt-4o with the proper prompts and parameters
    tasks = []
    count = 0
    for (q, q_id), a in q_and_as.items():
        messages = [
            {"role": "system", "content": scoring_prompt},
            {"role": "user", "content": user_prompt.format(question=q, student_response=llama_answers[count], correct_answers=a)}
        ]

        custom_id = q_id
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 100,
                "response_format": {"type": "json_schema",
                                    "json_schema": {
                                        "name": "score_response",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "explanation": {
                                                    "type": "string",
                                                },
                                                "score": {
                                                    "type": "boolean",
                                                }
                                            },
                                            "required": ["explanation", "score"],
                                            "additionalProperties": False
                                        },
                                    }
                                    },
                "messages": messages
            }
            }
        tasks.append(task)
        count += 1

    print(tasks[0])

    # Dump the llama model's answers into a jsonl file for the batch
    with open("../llama_score.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')
    print(f'Batch input file created.')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to openai
    batch_file = client.files.create(
        file=open("../llama_score.jsonl", 'rb'),
        purpose='batch'
    )
    batch_id = batch_file.id
    print(f'Batch file uploaded successfully. File ID: {batch_id}')

    # Submit the batch job to the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_job_id = batch_job.id
    print(f'Batch job submitted. Batch job ID: {batch_job_id}')

    # Check status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job_id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        if check.status == 'failed':
            print(f'Batch failed! {check}')
            exit()
        time.sleep(30)
    print("Done processing batch.")

    # Get the results back from GPT-4o
    results = []
    result_bytes = client.files.content(check.output_file_id).content
    result_str = result_bytes.decode('utf-8')
    print(result_str)  # Debugging: Check the raw response

    for line in result_str.splitlines():
        data = json.loads(line)
        content_str = data["response"]["body"]["choices"][0]["message"]["content"].strip()

        # Parse the content as JSON
        content_json = json.loads(content_str)

        # Extract the score
        score = content_json.get("score", False)  # Default to False if missing
        results.append(score)

    print(results)
    # Count correct answers
    count_correct = sum(1 for score in results if score)

    print(f"LLama's accuracy was {(count_correct/len(results))*100}")
    return ((count_correct/len(results))*100)

if __name__ == '__main__':
    score_mini()