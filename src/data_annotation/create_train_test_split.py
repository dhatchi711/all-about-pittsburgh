import json
import os


def make_prompt(context, question):
    prompt = f"You are a question-answering machine. Your task is to return a json object. \
            Do NOT give any explanation and you do not need to answer in full english sentences. \
            ONLY answer the question on the context given. \n\
                Given the context: {context} \n\
                    The question is: {question}"
    return prompt


def create_splits(file_list, output_path=None):
    combined_data = []

    for file in file_list:
        with open(file, 'r') as f:
            all_data = json.load(f)
            for data in all_data:
                if not isinstance(data, dict):
                    raise ValueError("Data should be a dictionary")
                try:
                    input = make_prompt(data['context'], data['question'])
                    output = data['answer']
                except:
                    continue
                json_obj = {
                    'input': input,
                    'output': output
                }
                combined_data.append(json_obj)

    # split the combined data into 90-10 split
    train_data = combined_data[:int(0.9 * len(combined_data))]
    test_data = combined_data[int(0.9 * len(combined_data)):]
    print(f"Train data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")

    with open(os.path.join(output_path, 'train.json'), 'w') as out_f:
        json.dump(train_data, out_f, indent=4)

    with open(os.path.join(output_path, 'test.json'), 'w') as out_f:
        json.dump(test_data, out_f, indent=4)



if __name__ == '__main__':
    file_list = ['../../data/annotated_data/final_annotated_data.json']
    create_splits(file_list, output_path='../../data/annotated_data')
