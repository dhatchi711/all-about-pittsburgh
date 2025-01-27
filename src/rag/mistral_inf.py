import torch
import re
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

def create_pipeline():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    pipe = pipeline(task="text-generation", model=model, max_new_tokens=200, tokenizer=tokenizer, device = "cuda")

    return pipe


def get_answer(pipe, question, context):
    prompt = [
        {"role": "user", "content": "You are a question-answering machine. Your task is to return a json object. \
            Do NOT give any explanation and you do not need to answer in full english sentences. \
            ONLY answer the question on the context given. \
            If you don't know the answer, say 'I do not know'."},

        {"role": "assistant", "content": "Understood, I will exactly follow the instructions and answer the question accordingly."},

        {"role": "user", "content": "{\
                'context': 'France is renowned for its rich culture, diverse landscapes, and historical significance, making it one of the most visited countries in the world. Paris, the capital city, is famous for its art, fashion, and architectural marvels like the Eiffel Tower and Notre-Dame Cathedral. From the scenic vineyards of Bordeaux to the sunlit beaches of the French Riviera, France offers a unique charm and beauty at every corner.'\
                'question': 'What is the capital of France?',\
                'answer': ''\
            }"},

        {"role": "assistant", "content": "{\
                'question': 'What is the capital of France?',\
                'answer': 'Paris'\
            }"},

        {"role": "user", "content": f"{{ \
                'context': '{context}',\
                'question': '{question}',\
                'answer': '',\
            }}"}
    ]
    
    result = pipe(prompt)
    json_str = result[-1]['generated_text'][-1]['content']
    # convert to json
    json_str = json_str.strip()
    # # remove all whitespaces more than one
    json_str = " ".join(json_str.split())
    # remove the whitespaces between the first { and the last }
    json_str = json_str.replace("{ ", "{")
    json_str = json_str.replace(" }", "}")
    # convert to json
    try:
        try:
            result = eval(json_str)
        except:
            json_str = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "", json_str)
            json_str = json_str.replace("s'", "s")
            json_str = re.sub(r'[“”]', '', json_str)
            json_str = re.sub(r"(?<![a-zA-Z])'((?!question|answer)\w+)'(?![a-zA-Z])", r"\1", json_str)
            if json_str[-1] != "}":
                if json_str[-1] == "'":
                    json_str += "}"
                else:
                    json_str += "'}"
            elif json_str[-1] == '}' and json_str[-2] != "'":
                json_str = json_str[:-1] + "'}"
            result = eval(json_str)
        return result['answer']
    except:
        return "I do not know"


if __name__ == "__main__":
    pipe = create_pipeline()
    question = "What languages do the performers sing in Pittsbugh opera?"
    context = "The Pittsburgh Opera is an American opera company based in Pittsburgh, Pennsylvania. The Pittsburgh Opera performs in the Benedum Center, a former movie palace that was renovated and reopened in 1987. The Pittsburgh Opera performs in English, Italian, German, and French. The company's home is at the Pittsburgh Opera Headquarters and Education Center in the city's Strip District. Pittsburgh Opera was founded in 1939 as the Pittsburgh Opera Company, and its first performance was in 1940. The company's most recent performance was in 2019."
    answer = get_answer(pipe, question, context)
    print(answer)