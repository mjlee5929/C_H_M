import os
from tqdm import tqdm

import copy
import re
import json
from openai import OpenAI
from json.decoder import JSONDecodeError


def main(input_file, output_file_path):
    text_json = None
    with open(input_file, 'r', encoding='utf-8') as f:
        text_json = json.load(f)
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # for ith, doc in text_json.items():
    for ith, doc in tqdm(text_json.items(), total=len(text_json), desc="Processing documents"):
        try:
            _texts_prompt = f"""당신은 본문에서 하나의 질문으로 2개 이상을 물어보는 질문을 만드는 데 도움이 되는 어시스턴트입니다.
다음과 같이 본문을 통해 2개 이상의 정보를 물어보는 질문을 생성하는 예시입니다.

본문:
2024년 7월 19일, 미국의 사이버 보안 회사인 크라우드스트라이크가 제작한 보안 소프트웨어의 잘못된 업데이트로 인해 마이크로소프트 윈도우를 실행하는 수많은 컴퓨터와 가상 머신이 충돌했다. 전 세계의 기업과 정부는 정보 기술 역사상 최대 규모의 중단으로 묘사되는 사태로 인해 영향을 받았다.[1]
붕괴된 산업 중에는 항공사, 공항, 은행, 호텔, 병원, 제조, 주식 시장, 방송 등이 있었다. 긴급 전화번호, 웹사이트 등 정부 서비스도 영향을 받았다. 오류가 발견돼 당일 수정됐지만, 사고로 인해 항공 항공편이 계속 지연되고 전자 결제 처리에 문제가 발생하며 응급 서비스가 중단되는 등의 문제가 발생하고 있다.[2][3][4][5]
시아란 마틴(Ciaran Martin)의 추정에 따르면 이번 사고로 인해 수십억 파운드에 달하는 경제적 피해가 발생할 것으로 예상된다.
    
질문:
1. 역대급 중단이 발생한 일자와 그 원인은 무엇인가?
2. 중단 사태로 피해를 입은 분야들과 피해 금액은 얼마인가?

위에 예시와 같이 주어진 본문을 이용해 2개 이상의 정보를 물어보는 질문들을 만들어주세요.
본문의 길이에 따라 1~4개 만들어주세요.

본문:
{doc["answer"].strip()}

질문:
"""
            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."},
                    {"role": "user", "content": _texts_prompt}
                ],
                top_p=0.25,
            )

            res_dict = {
                str(ith): {
                    "id": doc["id"],
                    "questions": completion.choices[0].message.content,
                    "answer": doc["answer"].strip(),
                }
            }

            with open(f"{output_file_path}/{doc['id']}.json", "w", encoding='utf-8') as f:
                json.dump(res_dict, f, ensure_ascii=False, indent=4)
        except JSONDecodeError as e:
            print(f"###FAIL - JSON Error {doc['id']}: {str(e)}")
            continue
        except Exception as e:
            print(f"###FAIL - Unexpected Error {doc['id']}: {str(e)}")
            continue



if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_file = "../data/kowiki_text.json" # kowiki파일을 id, answer의 dictionary 형태로 변환한 것
    output_file_path = "../data/kowiki_questions_added" # local model의 추론결과(생성한 질문)를 수집한 폴더
    os.makedirs(output_file_path, exist_ok=True)
    main(input_file, output_file_path)
