import streamlit as st
import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
from datetime import datetime

# 준지도
import torch
from transformers import GPT2Model, PreTrainedTokenizerFast, GPT2LMHeadModel
import faiss
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SimpleRNN, GRU, Dense


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





# GPT-2로 은행 관련 여부 판단 함수
def is_bank_related_gpt2(text):
    prompt = f"당신은 사용자가 말한 주제에 적합한지 판단하는 라벨러입니다. 이 텍스트는 은행,금융등에 관련된 내용과 관련이 있습니까?예, 아니요로 대답하시오. '{text}'"
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=950)
    try:
        outputs = headmodel.generate(inputs, max_length=inputs.shape[1] + 15, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "있습니다" in response or "예" in response
    except IndexError as e:
        st.write(f"Error: {e}, Input Text: {text}")
        return False

# GPT-2 모델에서 임베딩 추출 함수
def get_embeddings(input_ids):
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 모델 학습 및 예측을 위한 함수
def train_and_predict_model(model, X_labeled, y_labeled, X_unlabeled):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
    return (model.predict(X_unlabeled) > 0.5).astype(int)

# Streamlit 앱
st.title("GPT-2와 신경망을 활용한 텍스트 분류기")
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:", df.head())

    # 레이블링을 위한 샘플 추출
    labeling_df = pd.DataFrame(df['text'].sample(int(0.03 * len(df))))

    # 토크나이저 및 모델 불러오기
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    model = GPT2Model.from_pretrained("skt/kogpt2-base-v2")
    headmodel = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    MAX_TOKENS = 1024

    # GPT-2를 사용해 데이터 레이블링 및 임베딩 추출
    new_data = [get_embeddings(tokenizer.encode(text, truncation=True, max_length=MAX_TOKENS)) for text in labeling_df['text']]
    embeddings = np.array(new_data)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    labeled_data = [(text, 1 if is_bank_related_gpt2(text) else 0) for text in labeling_df['text']]
    df_labeled = pd.DataFrame(labeled_data, columns=['original_text', 'Label'])
    df_merged = pd.merge(df, df_labeled, left_on='text', right_on='original_text', how='left').drop(columns=['original_text'])

    # LSTM과 RNN 학습을 위한 데이터 준비
    texts = df_merged['tok_text'].apply(eval).astype(str)
    labels = df_merged['Label']
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_seq_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)
    labeled_indices = labels.notna()
    unlabeled_indices = ~labeled_indices
    X_labeled = padded_sequences[labeled_indices]
    y_labeled = labels[labeled_indices].astype(int)
    X_unlabeled = padded_sequences[unlabeled_indices]

    # LSTM을 통한 학습 및 예측
    model_LSTM = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    LSTM_pseudo_labels = train_and_predict_model(model_LSTM, X_labeled, y_labeled, X_unlabeled)
    df_merged.loc[unlabeled_indices, 'LSTM_Label'] = LSTM_pseudo_labels.flatten()

    # RNN을 통한 학습 및 예측
    model_RNN = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
        SimpleRNN(128),
        Dense(1, activation='sigmoid')
    ])
    RNN_pseudo_labels = train_and_predict_model(model_RNN, X_labeled, y_labeled, X_unlabeled)
    df_merged.loc[unlabeled_indices, 'RNN_Label'] = RNN_pseudo_labels.flatten()

    # LSTM과 RNN의 예측 결과가 다를 경우, GRU를 사용하여 재분류
    different_indices = df_merged[(df_merged['LSTM_Label'] != df_merged['RNN_Label']) & (unlabeled_indices)].index
    if different_indices.any():
        X_different = padded_sequences[different_indices]
        model_GRU = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
            GRU(128),
            Dense(1, activation='sigmoid')
        ])
        model_GRU.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_GRU.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
        gru_pseudo_labels = (model_GRU.predict(X_different) > 0.5).astype(int)
        df_merged.loc[different_indices, 'GRU_Label'] = gru_pseudo_labels.flatten()
        df_merged['final_label'] = np.where(df_merged['LSTM_Label'] == df_merged['RNN_Label'], df_merged['LSTM_Label'], df_merged['GRU_Label'])
    else:
        df_merged['final_label'] = df_merged['LSTM_Label']
    
    df_merged = df_merged[(df_merged['final_label'] != 0) & (df_merged['final_label'].notna())]
    st.write("레이블링된 데이터:", df_merged)
    st.download_button(label="CSV로 다운로드", data=df_merged.to_csv(index=False).encode('utf-8'), file_name='labeled_data.csv', mime='text/csv')
