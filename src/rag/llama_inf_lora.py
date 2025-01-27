import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def create_pipeline():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    new_model = "../../finetune/lora-llama3.1-finetune"
    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    pipe = pipeline(task="text-generation", model=model, max_new_tokens=200, temperature = 0.5, tokenizer=tokenizer, device = "cuda")

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
    return result[-1]['generated_text'][-1]['content']


if __name__ == "__main__":
    pipe = create_pipeline()
    question = "What languages do the performers sing in Pittsbugh opera?"
    context = "The Pittsburgh Opera is an American opera company based in Pittsburgh, Pennsylvania. The Pittsburgh Opera performs in the Benedum Center, a former movie palace that was renovated and reopened in 1987. The company's home is at the Pittsburgh Opera Headquarters and Education Center in the city's Strip District. Pittsburgh Opera was founded in 1939 as the Pittsburgh Opera Company, and its first performance was in 1940. The company's most recent performance was in 2019. The Pittsburgh Opera performs in English, Italian, German, and French."

    answer = get_answer(pipe, question, context)
    print(answer)