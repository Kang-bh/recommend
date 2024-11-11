import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import data_process

df = pd.read_csv("./book/reviews.csv")
books_df = pd.read_csv("./book/book_metadata.csv")

train_data = df[:1000]  # 데이터 크기를 3600개로 줄임
valid_data = df[3600:4800]
test_data = df[4800:6000]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

bert_model = bert_model.to(device)
gpt_model = gpt_model.to(device)

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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


def generate_gpt_prompt(review_embedding, recommended_description):
    prompt = (
        f"Based on the user's review embedding, recommend a book with the following description:\n"
        f"- Recommended book description: {recommended_description}.\n"
        f"- Review embedding: {review_embedding[:10]}."
        F"- Recommend Book : "
    )

    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)

    output = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1)

    generated_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

def recommend_book(review_embedding, book_embeddings, book_titles, book_descriptions):
    similarities = cosine_similarity(review_embedding.reshape(1, -1), book_embeddings)
    recommended_index = np.argmax(similarities)  # 가장 유사한 책 인덱스
    return book_titles[recommended_index], book_descriptions[recommended_index]  # 추천된 책 제목과 설명 반환



train_embeddings = get_bert_embeddings(bert_model, train_loader, device)

book_titles = books_df['Title'].tolist()  # 전체 책 제목 리스트
book_descriptions = books_df['description'].tolist()  # 전체 책 설명 리스트
book_embeddings = get_book_embeddings(bert_model, book_titles, bert_tokenizer, device)


print(f"start recommend")
first_embedding = train_embeddings[0]  # 첫 번째 임베딩
recommended_book_title, recommended_book_description = recommend_book(first_embedding, book_embeddings, book_titles, book_descriptions)  # 책 추천
recommended_text = generate_gpt_prompt(first_embedding, recommended_book_description)  # GPT를 통한 추천 텍스트 생성

print("Recommended Book Title:", recommended_book_title)  # 추천된 책 제목 출력
print("Recommended Book Description:", recommended_book_description)  # 추천된 책 설명 출력
print("Generated Recommendation from GPT:", recommended_text)  # GPT 생성 텍스트 출력