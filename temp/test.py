import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
ipmo

# 1. 리뷰 데이터 로드 (CSV 파일에서 불러오기)
df_reviews = pd.read_csv('reviews.csv')

# 2. 데이터 나누기
# 먼저 훈련+검증 데이터와 테스트 데이터를 80:20 비율로 나눔
train_val_df, test_df = train_test_split(df_reviews, test_size=0.2, random_state=42)

# 그다음 훈련+검증 데이터를 다시 80:20 비율로 나눠서 훈련 데이터와 검증 데이터를 만듦
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

# 데이터셋 크기 확인
print(f"훈련 데이터: {len(train_df)}개")
print(f"검증 데이터: {len(val_df)}개")
print(f"테스트 데이터: {len(test_df)}개")

# 3. BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 4. 리뷰 임베딩 생성 함수
def get_review_embedding(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].numpy()

# 5. 훈련, 검증, 테스트 데이터에 대해 임베딩 생성
train_df['embedding'] = train_df['review/text'].apply(get_review_embedding)
val_df['embedding'] = val_df['review/text'].apply(get_review_embedding)
test_df['embedding'] = test_df['review/text'].apply(get_review_embedding)

# 6. 책별로 리뷰 임베딩 평균 계산
train_book_embeddings = train_df.groupby('Title')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))
val_book_embeddings = val_df.groupby('Title')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))
test_book_embeddings = test_df.groupby('Title')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))

# 7. 유사도 계산 함수 (훈련 데이터에서 유사한 책 찾기)
def find_similar_books(book_title, book_embeddings, top_n=5):
    if book_title not in book_embeddings:
        print(f"'{book_title}'에 대한 정보가 없습니다.")
        return
    
    target_embedding = book_embeddings[book_title].reshape(1, -1)
    
    similarities = {}
    for title, embedding in book_embeddings.items():
        if title != book_title:
            sim = cosine_similarity(target_embedding, embedding.reshape(1, -1))[0][0]
            similarities[title] = sim
    
    similar_books = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return similar_books

# 8. 훈련 데이터로 유사한 책 찾기
recommended_books = find_similar_books("The Great Gatsby", train_book_embeddings)
print("추천 책 목록:")
for book, similarity in recommended_books:
    print(f"책 제목: {book}, 유사도: {similarity:.4f}")

# 9. 테스트 데이터로 최종 평가 (테스트 데이터에서 유사한 책 찾기)
test_recommended_books = find_similar_books("The Great Gatsby", test_book_embeddings)
print("\n테스트 데이터에서 추천 책 목록:")
for book, similarity in test_recommended_books:
    print(f"책 제목: {book}, 유사도: {similarity:.4f}")