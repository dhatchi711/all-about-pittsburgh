from prompt import LLM_FEW_SHOT_PROMPT
import argparse
import re
from tqdm import tqdm

def escape_special_chars(text):
        """Escapes special characters like \n and \t in the text."""
        return text.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t')


def preprocess_question(question):
    question = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "", question)
    question = question.replace("s'", "s")
    question = re.sub(r'[“”]', '', question)
    question = re.sub(r"(?<![a-zA-Z])'((?!question|answer)\w+)'(?![a-zA-Z])", r"\1", question)
    return question

def get_questions(file_path):
    '''Get questions from the input file

    Args:
        file_path (str): path to the input file

    Returns:
        list: list of questions
    '''
    questions = []
    
    if ".txt" in file_path:
        with open(file_path, 'r') as file:
            for line in file:
                line = preprocess_question(line)
                questions.append(line)
    elif ".csv" in file_path:
        import pandas as pd
        df = pd.read_csv(file_path)
        # preprocess questions
        df['question'] = df['question'].apply(preprocess_question)
        questions = df['question'].tolist

    else:
        raise ValueError("File format not supported")
    
    return questions


def setup_and_inference(model_name, questions):
    '''set up model for generations

    Args:
        model_name (str): name of the model
    
    Returns:
        model: model for generation
        pipe: pipeline for question answering
    '''
    if model_name not in ["meta-llama/Llama-3.1-8B-Instruct", "deepset/roberta-base-squad2", "mistralai/Mistral-7B-Instruct-v0.1"]:
        raise ValueError("Model name not supported")

    if model_name == "meta-llama/Llama-3.1-8B-Instruct":
        from llama_inf import create_pipeline, get_answer
    elif model_name == "deepset/roberta-base-squad2":
        from roberta_inf import create_pipeline, get_answer
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        from mistral_inf import create_pipeline, get_answer
    else:
        raise ValueError("Model name not supported")

    pipe = create_pipeline()

    answers = [] 
    for question in tqdm(questions, desc="Processing Questions"):
        contexts = " "
        answer = get_answer(pipe, question, contexts)
        answer = escape_special_chars(answer)
        answers.append(answer)

    return answers    

def write_output(file_path, answers):
    with open(file_path, 'w') as file:
        for answer in answers:
            file.write(answer + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help='HuggingFace model name')
    parser.add_argument("test_input", type=str, help='path to the input .txt or .csv file')
    parser.add_argument("output", type=str, 
                        help='path to output .txt file to which the answers should be written')
    args = parser.parse_args()

    # file_path = "../../data/test_data/questions.txt"
    input_filepath = args.test_input

    model_name = args.model_name
    output_filepath = args.output

    
    questions = get_questions(input_filepath)
    answers = setup_and_inference(model_name, questions)

    write_output(output_filepath, answers)