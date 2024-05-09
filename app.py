import streamlit as st
import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
from datetime import datetime



# 파일 업로드 기능

uploaded_file = st.file_uploader("파일을 선택해주세요.", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("파일이 성공적으로 업로드되었습니다.")
    st.write(df.head())
    st.wirte(df.info())# 데이터 미리보기

    # 데이터 전처리 시작
    with st.spinner("데이터를 전처리 중입니다..."):
        # 필요없는 열 제거
        df = df[['keyword', 'tit', 'body', 'comment', 'date']]

        # 중복값 제거
        df = df.drop_duplicates(subset=['tit', 'body'])

        # 결측치 처리
        df['body'] = df['body'].fillna('')

        # 컬럼 합치기
        df.loc[:, 'text'] = df['tit'] + ' ' + df['body'] + ' ' + df['comment']
        df = df.drop(['tit', 'body', 'comment'], axis=1)

        # 전처리 함수
        def preprocess_text(text):
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[0-9]+갤', '개월', text)
            text = re.sub(r'[^가-힣.]', ' ', text)
            text = re.sub(r'\.\s+', '.', text)
            text = re.sub(r'\.{2,}', '.', text)
            text = re.sub(r'\s+', ' ', text)
            return text
        
        df.loc[:, 'text'] = df['text'].apply(preprocess_text)

        # 형태소 분석기 설정
        def get_tokenizer(tokenizer_name):
            if tokenizer_name == "komoran":
                return Komoran()
            elif tokenizer_name == "okt":
                return Okt()
            elif tokenizer_name == "mecab":
                return Mecab()
            elif tokenizer_name == "hannanum":
                return Hannanum()
            elif tokenizer_name == "kkma":
                return Kkma()
            else:
                return Okt()

        tokenizer = get_tokenizer("okt")
        
        def pos_tagging_and_filter(text):
            pos_tokens = tokenizer.pos(text)
            filtered_tokens = [token for token, pos in pos_tokens if pos != 'Josa']
            return filtered_tokens
        
        df['tok_text'] = df['text'].apply(pos_tagging_and_filter)

        # 형태소 분석 결과를 CSV 파일로 저장
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M%S")
        filename = f"형태소분석_{formatted_time}.csv"
        df.to_csv(filename, index=False)

        # 완료 메시지 및 파일 다운로드 링크
        st.success('데이터 전처리가 완료되었습니다.')
        st.download_button('결과 다운로드', data=df.to_csv(index=False), file_name=filename, mime='text/csv')

# 스트림릿 앱 실행
if __name__ == '__main__':
    st.title('한국어 자연어 데이터 전처리')
    st.write('데이터를 업로드하고 전처리를 진행하세요.')


