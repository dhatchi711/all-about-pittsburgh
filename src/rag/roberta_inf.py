from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

def create_pipeline():
    pipe = pipeline('question-answering', model=model_name, max_new_tokens=50, tokenizer=model_name, device="cuda")
    return pipe

def get_answer(pipe, question, context):
    res = pipe(question=question, context=context)
    return res['answer']

if __name__ == "__main__":
    pipe = create_pipeline()
    question = "Why is model conversion important?"
    context = "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."
    answer = get_answer(pipe, question, context)
    print(answer)