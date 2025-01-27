from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from prompt import LLM_FEW_SHOT_PROMPT
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
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
        df['Question'] = df['Question'].apply(preprocess_question)
        questions = df['Question'].tolist()

    else:
        raise ValueError("File format not supported")
    
    return questions

def setup_embedding_model(model_name):
    '''Set up embedding model

    Args:
        model_name (str): name of the model

    Returns:
        HuggingFaceBgeEmbeddings: embedding model
    '''
    model_kwargs = {
        'device': 'cpu', 
        'trust_remote_code':True
    }
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction = "search_query:",
        embed_instruction = "search_document:"
    )

    return embed_model

def get_retriever(retriever_name, embed_model):
    '''
    set up retriever

    Args:
        retriever_name (str): name of the retriever
        embed_model (HuggingFaceBgeEmbeddings): embedding model
    
    Returns:
        retriever: retriever
    '''
    if retriever_name not in ["vectordb", "bm25"]:
        raise ValueError("Retriever not supported")

    if retriever_name == 'vectordb':
        vectordb = FAISS.load_local('../../vector_store', embed_model, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(search_kwargs={'k': 4})
    elif retriever_name == 'bm25':
        directory_path = "../../data/scraped_data"
        docs = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".txt"): 
                    file_path = os.path.join(root, file)
                    loader = TextLoader(file_path)
                    docs.extend(loader.load())  

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=128, add_start_index=True
            )
        all_splits = text_splitter.split_documents(docs)

        retriever = BM25Retriever.from_documents(all_splits)
        retriever.k = 4

    return retriever

def setup_and_inference(model_name, retriever_name, retriever, questions):
    '''set up model for generations

    Args:
        model_name (str): name of the model
    
    Returns:
        model: model for generation
        pipe: pipeline for question answering
    '''
    model_name = sys.argv[1]
    if model_name not in ["meta-llama/Llama-3.1-8B-Instruct", "deepset/roberta-base-squad2", "mistralai/Mistral-7B-Instruct-v0.1", "llama-3.1-8B-Instruct-finetuned"]:
        raise ValueError("Model name not supported")

    if model_name == "meta-llama/Llama-3.1-8B-Instruct":
        from llama_inf import create_pipeline, get_answer
    elif model_name == "deepset/roberta-base-squad2":
        from roberta_inf import create_pipeline, get_answer
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        from mistral_inf import create_pipeline, get_answer
    elif model_name == "llama-3.1-8B-Instruct-finetuned":
        from llama_inf_lora import create_pipeline, get_answer
    else:
        raise ValueError("Model name not supported")

    pipe = create_pipeline()

    answers = [] 
    for question in tqdm(questions, desc="Processing Questions"):
        if retriever_name == 'faiss':
            question = "search_query: " + question
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs[:4]]
        contexts = " ".join(contexts)
        
        try:
            answer = get_answer(pipe, question, contexts)
        except:
            print(f"Error in getting answer for {question}")
            answer = "I do not know"
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
    parser.add_argument("retriever_name", type=str, help='either bm25 or faiss')
    parser.add_argument("test_input", type=str, help='path to the input .txt or .csv file')
    parser.add_argument("output", type=str, 
                        help='path to output .txt file to which the answers should be written')
    args = parser.parse_args()

    # file_path = "../../data/test_data/questions.txt"
    input_filepath = args.test_input

    model_name = args.model_name
    retriever_name = args.retriever_name
    output_filepath = args.output

    
    questions = get_questions(input_filepath)

    embedding_model = "nomic-ai/nomic-embed-text-v1"
    embed_model = setup_embedding_model(embedding_model)

    retriever = get_retriever(retriever_name, embed_model)

    answers = setup_and_inference(model_name, retriever_name, retriever, questions)

    write_output(output_filepath, answers)