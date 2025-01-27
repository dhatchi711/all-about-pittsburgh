import re

def extract_context_and_questions(input_file, context_file, question_file):
    pattern = re.compile(
        r'Given the context:\s*(.*?)\s*The question is:\s*(.*?)\s*$',
        re.DOTALL
    )
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(context_file, 'w', encoding='utf-8') as ctxfile, \
         open(question_file, 'w', encoding='utf-8') as qfile:
        
        for line in infile:
            match = pattern.search(line)
            if match:
                context = match.group(1).strip()
                question = match.group(2).strip()
                
                ctxfile.write(context + '\n')
                qfile.write(question + '\n')

if __name__ == "__main__":
    input_file = '../../data/test_data/questions_with_context.txt'
    context_file = '../../data/test_data/context.txt'
    questions_file = '../../data/test_data/questions.txt'

    extract_context_and_questions(input_file, context_file, questions_file)
