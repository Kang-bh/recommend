import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import data_process
import time
from evaluate import hit_ratio, ndcg_at_k

current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_dir:", current_dir)

start_time = time.time()

df = pd.read_csv(
    current_dir + "/book/reviews.csv",
    encoding='utf-8',
    on_bad_lines='skip',
    engine="python",
    nrows=6400
)
books_df = pd.read_csv(
    current_dir + "/book/book_metadata.csv", 
    encoding='utf-8',
    on_bad_lines='skip',
    engine="python"
)

print("Completed reading CSVs in {:.2f} seconds".format(time.time() - start_time))

# 데이터 분할
train_data = df[:1000]
valid_data = df[3600:4800]
test_data = df[4800:6000]

print("Completed splitting data in {:.2f} seconds".format(time.time() - start_time))

# 디바이스 설정
print("torch available : ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Completed device initialization in {:.2f} seconds".format(time.time() - start_time))

# 모델 및 토크나이저 로딩
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

bert_model = bert_model.to(device)
gpt_model = gpt_model.to(device)

print("Completed model loading in {:.2f} seconds".format(time.time() - start_time))

# 데이터셋 및 데이터로더 생성
class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = str(self.data.iloc[index]['review/text'])
        book_title = str(self.data.iloc[index]['Title'])
        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'book_title': book_title
        }

train_dataset = ReviewDataset(train_data, bert_tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

print("Completed dataset and dataloader creation in {:.2f} seconds".format(time.time() - start_time))

# BERT 임베딩 생성 함수
def get_bert_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)

# 책 임베딩 생성 함수
def get_book_embeddings(model, book_titles, tokenizer, device):
    embeddings = []
    for title in book_titles:
        if pd.isna(title):
            continue
        
        encoding = tokenizer.encode_plus(
            title,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)

    return np.vstack(embeddings)

# GPT 프롬프트 생성 함수
def generate_gpt_prompt(review_embedding, recommended_description):
    prompt = (
        f"Based on the user's review embedding, recommend a book with the following description:\n"
        f"- Recommended book description: {recommended_description}.\n"
        f"- Review embedding: {review_embedding[:10]}."
        f"- Recommend Book : "
    )

    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 패딩 토큰이 설정되었는지 확인하고 필요 시 설정
    if gpt_tokenizer.pad_token_id is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # 패딩 토큰을 EOS로 설정

    # Attention mask 생성
    attention_mask = (input_ids != gpt_tokenizer.pad_token_id).long()

    output = gpt_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50)

    generated_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


# 책 추천 함수
def recommend_book(review_embedding, book_embeddings, book_titles, book_descriptions):
    similarities = cosine_similarity(review_embedding.reshape(1, -1), book_embeddings)
    recommended_index = np.argmax(similarities)  # 가장 유사한 책 인덱스
    return book_titles[recommended_index], book_descriptions[recommended_index]  # 추천된 책 제목과 설명 반환

# 임베딩 생성 및 추천 과정 시간 측정 시작
embedding_start_time = time.time()

print("Start train embeddings")
train_embeddings = get_bert_embeddings(bert_model, train_loader, device)
print("Completed train embeddings in {:.2f} seconds".format(time.time() - embedding_start_time))

book_titles = books_df['Title'].tolist()
book_descriptions = books_df['description'].tolist()
book_embeddings_start_time = time.time()
book_embeddings = get_book_embeddings(bert_model, book_titles, bert_tokenizer, device)
print("Completed book embeddings in {:.2f} seconds".format(time.time() - book_embeddings_start_time))

recommendation_start_time = time.time()
print(f"Start recommend")
first_embedding = train_embeddings[0]
recommended_book_title, recommended_book_description = recommend_book(first_embedding, book_embeddings, book_titles, book_descriptions)
recommended_text_start_time = time.time()
recommended_text = generate_gpt_prompt(first_embedding, recommended_book_description)
print("Completed recommendation and text generation in {:.2f} seconds".format(time.time() - recommendation_start_time))

print("Recommended Book Title:", recommended_book_title)
print("Recommended Book Description:", recommended_book_description)
print("Generated Recommendation from GPT:", recommended_text)

# 전체 실행 시간 출력
total_time_end_time = time.time()
print("Total execution time: {:.2f} seconds".format(total_time_end_time - start_time))


# 평가 데이터셋 생성
valid_dataset = ReviewDataset(valid_data, bert_tokenizer, max_len=128)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 검증 데이터에 대한 임베딩 생성
valid_embeddings = get_bert_embeddings(bert_model, valid_loader, device)

# 평가 함수
def evaluate_recommendations(embeddings, book_embeddings, book_titles, actual_titles, k=10):
    ndcg_scores = []
    hit_ratios = []
    
    for i, embedding in enumerate(embeddings):
        similarities = cosine_similarity(embedding.reshape(1, -1), book_embeddings)
        recommended_indices = np.argsort(similarities[0])[::-1][:k]
        
        recommended_titles = [book_titles[idx] for idx in recommended_indices]
        
        # NDCG 계산
        relevance = [1 if title in actual_titles[i] else 0 for title in recommended_titles]
        ndcg_scores.append(ndcg_at_k(relevance, k))
        
        # Hit Ratio 계산
        hit_ratios.append(hit_ratio(recommended_titles, [actual_titles[i]]))
    
    return np.mean(ndcg_scores), np.mean(hit_ratios)

# 평가 실행
print(f"start evaluation")
actual_titles = valid_data['Title'].tolist()
ndcg, hr = evaluate_recommendations(valid_embeddings, book_embeddings, book_titles, actual_titles)

print(f"NDCG@10: {ndcg:.4f}")
print(f"Hit Ratio@10: {hr:.4f}")
