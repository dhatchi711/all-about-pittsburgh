import os
import json
from meta_ai_api import MetaAI
from tqdm import tqdm

bad_docs = []

def generate_question_answer_pairs(file_content, file_path):
    # extract first 400k characters of the file content
    document = file_content[:400000]

    prompt = f'''
        Instructions: You are training a question-answering machine. Your task is to return a JSON object of 5 questions per document with 3 fields: context, question, answer.
        Do NOT give any explanation in the answer and you do not need to answer in full English sentences.
        ONLY answer the question on the context given.
        If you do not know the answer, return 'I do not know'.

        Example:
        'context': 'France is renowned for its rich culture, diverse landscapes, and historical significance, making it one of the most visited countries in the world. Paris, the capital city, is famous for its art, fashion, and architectural marvels like the Eiffel Tower and Notre-Dame Cathedral.',
        'question': 'What is the capital of France?',
        'answer': 'Paris'.

        Document:
        {document}

        *Output Format*:

        [
            {{
                "context": "<insert 2-3 sentences here>",
                "question": "<insert question here>",
                "answer": "<insert concise answer here>"
            }},
            {{
                "context": "<insert 2-3 sentences here>",
                "question": "<insert question here>",
                "answer": "<insert concise answer here>"
            }},
            {{
                "context": "<insert 2-3 sentences here>",
                "question": "<insert question here>",
                "answer": "<insert concise answer here>"
            }},
            {{
                "context": "<insert 2-3 sentences here>",
                "question": "<insert question here>",
                "answer": "<insert concise answer here>"
            }},
            {{
                "context": "<insert 2-3 sentences here>",
                "question": "<insert question here>",
                "answer": "<insert concise answer here>"
            }}
        ]
    '''
    try: 
        response = ai.prompt(message=prompt)
        message = response.get('message', '[]')
    except: 
        print(f"Error processing file: {file_path}")
        bad_docs.append(file_path)
        return []

    try:
        question_answer_context = json.loads(message)
        return question_answer_context
    except json.JSONDecodeError:
        print("Error decoding JSON from the response.")
        return []


def process_directory(input_directory):
    all_qapairs = []

    for index, filename in enumerate(tqdm(os.listdir(input_directory), desc="Processing files")):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            print(f"Processing file: {filename}")
            question_answer_pairs = generate_question_answer_pairs(file_content, file_path)

            for pair in question_answer_pairs:
                context = pair.get("context", "No context")
                question = pair.get("question", "No question")
                answer = pair.get("answer", "No answer")
                all_qapairs.append({'Context': context, 'Q': question, 'A': answer})

    return all_qapairs


if __name__ == "__main__":
    print('here')
    ai = MetaAI()
    print("meta ai")

    inp_dir = os.path.join("../../", "data", "scraped_data")
    print(inp_dir)
    out_dir = os.path.join("../../", "data", "annotated_data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # list of directories/webpages
    # webpages = [f.name for f in os.scandir(inp_dir) if f.is_dir()]
    webpages = [
        "bananasplitfest.com",
        "carnegiemuseums.org",
        "downtownpittsburgh.com",
        "events.cmu.edu",
        "littleitalydays.com",
        "pittsburgh.events",
        "pittsburghopera.org",
        "pittsburghrestaurantweek.com",
        "trustarts.org",
        "www.britannica.com",
        "www.cmu.edu",
        "www.heinzhistorycenter.org",
        "www.mlb.com",
        "www.nhl.com"
    ]

    for webpage in webpages:
        dir = os.path.join(inp_dir, webpage)

        all_qapairs = process_directory(dir)

        output_filename = os.path.join(out_dir, f"{webpage}.jsonl")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for pair in all_qapairs:
                json.dump(pair, f, ensure_ascii=False)
                f.write('\n')

        print(f"Saved all Q&A pairs to: {output_filename}")

    with open('../../data/bad_docs.txt', 'w') as file:
        for url in bad_docs:
            file.write(f"{url}\n")