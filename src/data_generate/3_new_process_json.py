import json
import re
import copy
import os

from tqdm import tqdm




def remove_similar_questions(questions, threshold=0.7):
    def word_similarity(q1, q2):
        words1 = set(q1.split())
        words2 = set(q2.split())
        union_size = len(words1.union(words2))
        # 빈 집합 체크
        if union_size == 0:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    unique_questions = []
    for q in questions:
        if not any(word_similarity(q, uq) > threshold for uq in unique_questions):
            unique_questions.append(q)
        else:
            print(f"Similar question found: {q} - {unique_questions}")
    return unique_questions

def parse_questions(json_file):
    ret = {}
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data.values():
        questions = item['questions']
        parsed_questions = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)', questions, re.DOTALL)
        
        print(f"ID: {item['id']}")
        one_passage = copy.deepcopy(item)
        
        list_questions = []
        for i, q in enumerate(parsed_questions, 1):
            print(f"Question {i}: {q.strip()}")
            list_questions.append(q.strip())
        new_list_questions = remove_similar_questions(list_questions)
        
        one_passage["list_questions"] = new_list_questions
        
        ret[item['id']] = one_passage
    return ret

def process_all_files(input_path, output_file):
    result = {}
    
    for filename in tqdm(os.listdir(input_path), desc="Processing files"):
        if filename.endswith('.json'):
            file_path = os.path.join(input_path, filename)
            result.update(parse_questions(file_path))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_path = "../data/kowiki_questions_added" # local model의 추론결과(생성한 질문)를 수집한 폴더
    output_file = "../data/kowiki_questions.json" # 생성한 질문들을 전처리하고 하나의 파일로 통합
    process_all_files(input_path, output_file)
    