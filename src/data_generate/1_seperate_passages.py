import json
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter



def main(input_file, output_file, SEP_TOKEN):
    # 텍스트 파일 로드
    loader = TextLoader(input_file)
    documents = loader.load()
    
    # 대용량 문서를 분할하기 위한 splitter (1차 분할)
    doc_splitter = CharacterTextSplitter(
        separator=SEP_TOKEN, 
        chunk_size=200000,  # 최대 청크 크기
        chunk_overlap=0     # 청크 간 중복 없음
    )
    docs = doc_splitter.split_documents(documents)

    # 더 작은 단위로 텍스트 분할을 위한 splitter (2차 분할)
    text_splitter = CharacterTextSplitter(
        separator=SEP_TOKEN,
        chunk_size=1000,    # 최대 청크 크기 1000자
        chunk_overlap=0,     
        length_function=len  # 길이 계산 함수
    )

    # 결과를 저장할 딕셔너리
    res_dict = {}
    doc_id = 0
    
    # 각 문서 처리
    for n_doc, doc in tqdm(enumerate(docs), total=len(docs), desc="Processing documents"):
        # 텍스트를 더 작은 단위로 분할
        _texts_split = text_splitter.split_text(doc.page_content)

        for texts in _texts_split:
            # 구분자가 있으면 추가 분할
            if SEP_TOKEN in texts:
                texts_list = texts.split(SEP_TOKEN)
            else:
                texts_list = [texts]

            # 각 텍스트 조각 처리
            for text_doc in texts_list:
                # 200자 이상인 텍스트만 처리
                if len(text_doc) > 200:
                    # 1000자 제한
                    if len(text_doc) > 1000:
                        _texts_doc = text_doc[0:1000].strip()
                    else:
                        _texts_doc = text_doc
                    
                    # 결과 딕셔너리에 저장
                    doc_id += 1
                    res_dict[str(doc_id)] = {
                        "id": str(doc_id),
                        "answer": _texts_doc.strip(),
                    }
    
    # JSON 파일로 저장
    with open(output_file, "w") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_file = "../data/kowiki_dump.txt" # kowiki dump 파일
    output_file = "../data/kowiki_text.json" # kowiki파일을 id, answer의 dictionary 형태로 변환
    SEP_TOKEN = "\n\n\n\n"

    main(input_file, output_file, SEP_TOKEN)

                       