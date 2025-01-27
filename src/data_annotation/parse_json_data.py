import json

dirs = ['train_data', 'test_data']

def escape_special_chars(text):
        return text.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t')

def parse_json(input_file, questions_file, reference_answers_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(questions_file, 'w') as q_file, open(reference_answers_file, 'w') as a_file:
        for item in data:
            question = item['input']
            answer = item['output']
            
            escaped_question = escape_special_chars(question)
            escaped_answer = escape_special_chars(answer)
            
            q_file.write(escaped_question + '\n')            
            a_file.write(escaped_answer + '\n')

for dir in dirs:
    input_file = f'../../data/{dir}/{dir}.json'
    questions_file = f'../../data/{dir}/questions.txt'
    reference_answers_file = f'../../data/{dir}/reference_answers.txt'
    parse_json(input_file, questions_file, reference_answers_file)