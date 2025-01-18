import json
import os
import time

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser



def main(input_file="../data/kowiki_questions.json", output_file="../data/kowiki_train_questions.json", output_path=""):
    if output_path == "":
        output_path = output_file.replace(".json", "")
        os.makedirs(output_path, exist_ok=True)

    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries in korean related to: {question} \n
    expected output format = ["query1", ...]
    Output (2 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    
    # JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = {}
    limit_idx = 1500

    # 각 항목에 대해 list_questions 처리
    for item_id, item in data.items():
        if "decomposed_questions" in data[item_id]:
            output_data[item_id] = data[item_id]
            limit_idx -= 1
            continue
        
        if limit_idx < 1:
            break
        else:
            limit_idx -= 1
        list_questions = item['list_questions']
        
        decomposed_questions = {}
        for qid, question in enumerate(list_questions):
            try:
                result = generate_queries_decomposition.invoke({"question": question})
            
                if len(result) < 2:
                    continue
                decomposed_questions[qid] = {}
                decomposed_questions[qid]["question"] = question
                decomposed_questions[qid]["result"] = result

                # 결과 처리 (예: 출력)
                time.sleep(1)
            except Exception as e:
                print(f"Error processing question {question}: {str(e)}")
                time.sleep(3)
                continue

        output_data[item_id] = data[item_id]
        output_data[item_id]['decomposed_questions'] = decomposed_questions
        with open(f'{output_path}/{item_id}.json', 'w', encoding='utf-8') as wf:
            json.dump(output_data[item_id], wf, ensure_ascii=False, indent=4)

    with open(output_file, 'w', encoding='utf-8') as wf:
        json.dump(output_data, wf, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_file = "../data/kowiki_questions.json" # 생성한 질문들을 전처리하고 하나의 파일로 통합
    output_file = "../data/kowiki_train_questions.json" # openai 응답을 모은 최종결과 파일
    main(input_file, output_file)