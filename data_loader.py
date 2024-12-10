# import os
# import pandas as pd
# import torch
# from azure.ai.openai import AzureOpenAI
# from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, BOOKS_PATH, REVIEWS_PATH
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# def load_data(books_path, reviews_path, save_path="last_index.txt", output_path="embeddings.csv", num_books=6400):
#     # 책 데이터 로드 (6400개로 제한)
#     books_df = pd.read_csv(books_path).head(num_books)
    
#     # 리뷰 데이터 로드
#     reviews_df = pd.read_csv(reviews_path)
    
#     # 6400개의 책에 해당하는 리뷰만 필터링
#     reviews_df = reviews_df[reviews_df['asin'].isin(books_df['asin'])]
    
#     # 데이터 분할
#     train_data, test_data = train_test_split(reviews_df, test_size=0.2, random_state=42)
    
#     client = AzureOpenAI(
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_key=AZURE_OPENAI_API_KEY,
#         api_version="2023-05-15"
#     )
    
#     # GPU 사용 가능 여부 확인
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     last_index = 0
#     if os.path.exists(save_path):
#         with open(save_path, "r") as f:
#             last_index = int(f.read().strip())
    
#     embeddings = []
#     if os.path.exists(output_path):
#         embeddings_df = pd.read_csv(output_path)
#         embeddings = embeddings_df['embedding'].tolist()
#         books_df = books_df.merge(embeddings_df[['asin', 'embedding']], on='asin', how='left')
    
#     batch_size = 32  # 적절한 배치 크기 설정
    
#     try:
#         for i in tqdm(range(last_index, len(books_df), batch_size), desc="Generating embeddings"):
#             batch = books_df.iloc[i:i+batch_size]
#             texts = batch.apply(lambda row: f"{row['title']} {row['description']} {row['authors']} {row['categories']}", axis=1).tolist()
            
#             # Azure OpenAI API를 통해 임베딩 생성
#             batch_embeddings = client.embeddings.create(input=texts, model="text-embedding-ada-002").data
            
#             # 임베딩을 GPU로 이동
#             batch_embeddings_tensor = torch.tensor([emb.embedding for emb in batch_embeddings], device=device)
            
#             # 결과를 CPU로 이동하고 리스트에 추가
#             embeddings.extend(batch_embeddings_tensor.cpu().numpy())
            
#             # 현재 인덱스 저장
#             with open(save_path, "w") as f:
#                 f.write(str(i + len(batch)))
            
#             # 임베딩을 CSV 파일에 저장
#             pd.DataFrame({'asin': batch['asin'], 'title': batch['title'], 'embedding': embeddings[-len(batch):]}).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    
#     except Exception as e:
#         print(f"Error occurred at index {i}: {str(e)}")
#         print("Embedding generation stopped. You can resume later from the last saved index.")
    
#     books_df['text_embedding'] = embeddings
    
#     return books_df, train_data, test_data

# books_df, train_data, test_data = load_data(BOOKS_PATH, REVIEWS_PATH)
# print(f"Loaded {len(books_df)} books and {len(train_data) + len(test_data)} reviews")


import os
import pandas as pd
import torch
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, BOOKS_PATH, REVIEWS_PATH
import csv
import sys
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
import numpy as np



csv.field_size_limit(sys.maxsize)

def load_data(books_path, reviews_path, save_path="last_index.txt", output_path="embeddings.csv", num_books=6400):
    start_time = time.time()

    # Load book data
    books_load_start = time.time()
    books_df = pd.read_csv(books_path, on_bad_lines='skip', engine="python").head(num_books)
    books_load_end = time.time()
    print(f"Books data loading time: {books_load_end - books_load_start:.2f} seconds")

    # Load review data
    reviews_load_start = time.time()
    reviews_df = pd.read_csv(reviews_path, on_bad_lines='skip', engine="python")
    reviews_load_end = time.time()
    print(f"Reviews data loading time: {reviews_load_end - reviews_load_start:.2f} seconds")

    # Filter reviews
    filter_start = time.time()
    reviews_df = reviews_df[reviews_df['Title'].isin(books_df['Title'])]
    filter_end = time.time()
    print(f"Reviews filtering time: {filter_end - filter_start:.2f} seconds")

    # Split data
    split_start = time.time()
    train_data, test_data = train_test_split(reviews_df, test_size=0.2, random_state=42)
    split_end = time.time()
    print(f"Data splitting time: {split_end - split_start:.2f} seconds")

    # Initialize Sentence Transformer model
    model_init_start = time.time()
    model = SentenceTransformer('all-mpnet-base-v2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_init_end = time.time()
    print(f"Model initialization time: {model_init_end - model_init_start:.2f} seconds")
    print(f"Using device: {device}")

    last_index = 0
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            last_index = int(f.read().strip())

    embeddings = []
    if os.path.exists(output_path):
        embeddings_load_start = time.time()
        embeddings_df = pd.read_csv(output_path)
        embeddings = embeddings_df['embedding'].tolist()
        print(embeddings_df.head())
        print(books_df.head())

        books_df = books_df.merge(embeddings_df[['Title', 'embedding']], on='Title', how='left')
        embeddings_load_end = time.time()
        print(f"Existing embeddings loading time: {embeddings_load_end - embeddings_load_start:.2f} seconds")

    batch_size = 32

    embedding_generation_start = time.time()
    try:
        for i in tqdm(range(last_index, len(books_df), batch_size), desc="Generating embeddings"):
            batch = books_df.iloc[i:i+batch_size]
            texts = batch.apply(lambda row: f"{row['Title']} {row['description']} {row['authors']} {row['categories']}", axis=1).tolist()
            
            # Generate embeddings using Sentence Transformers
            batch_embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            
            # Move results to CPU and add to list
            embeddings.extend(batch_embeddings.cpu().numpy())
         
            # Save current index
            with open(save_path, "w") as f:
                f.write(str(i + len(batch)))
            
            # Save embeddings to CSV file
            # 임베딩을 문자열로 변환하여 저장
            embeddings_str = [' '.join(map(str, emb)) for emb in batch_embeddings.cpu().numpy()]
            pd.DataFrame({'Title': batch['Title'], 'embedding': embeddings_str}).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    
    except Exception as e:
        print(f"Error occurred at index {i}: {str(e)}")
        print("Embedding generation stopped. You can resume later from the last saved index.")
    
    embedding_generation_end = time.time()
    print(f"Embedding generation time: {embedding_generation_end - embedding_generation_start:.2f} seconds")

    # 여기에 새로운 코드 추가
    print(f"Number of books in books_df: {len(books_df)}")
    print(f"Number of generated embeddings: {len(embeddings)}")

    # 누락된 임베딩 처리
    if len(embeddings) < len(books_df):
        print(f"Warning: {len(books_df) - len(embeddings)} embeddings are missing.")
        default_embedding = [0] * 768  # 또는 적절한 기본값 사용
        while len(embeddings) < len(books_df):
            embeddings.append(default_embedding)

    # 임베딩을 books_df에 안전하게 추가
    if len(embeddings) == len(books_df):
        books_df['text_embedding'] = embeddings
    else:
        print(f"Warning: Mismatch in lengths. books_df: {len(books_df)}, embeddings: {len(embeddings)}")
        books_df['text_embedding'] = pd.Series(embeddings + [None] * (len(books_df) - len(embeddings)))

    end_time = time.time()
    print(f"Total data loading and processing time: {end_time - start_time:.2f} seconds")

    return books_df, train_data, test_data
    
def map_title_to_int(title, title_to_int):
    return title_to_int.get(title, 0)  # 매핑에 없는 제목은 0(패딩)으로 처리

def convert_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, str):
        return [float(i) for i in x.strip('[]').split()]
    else:
        return x  # 이미 리스트 형태인 경우

class BookSequenceDataset(Dataset):
    def __init__(self, sequences, embeddings):
        self.sequences = sequences
        self.embeddings = embeddings

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq, self.embeddings[seq]

def create_dataloader(data, books_df, batch_size, shuffle=True):
    print("Column names:")
    for col in data.columns:
        print(f"- {col}")

    print("\nColumn data types:")
    print(data.dtypes)

    print("\nNumber of non-null values in each column:")
    print(data.count())

    # Check if required columns exist
    required_columns = ['User_id', 'review/time', 'Title']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")

    if 'text_embedding' not in books_df.columns:
        raise ValueError("'text_embedding' column not found in books_df")

    # Create a mapping of unique titles to integers
    title_to_int = {title: i+1 for i, title in enumerate(books_df['Title'])}
    title_to_int['<PAD>'] = 0  # 패딩을 위한 특별한 토큰

    # 사용자별로 아이템 시퀀스 생성
    user_sequences = data.groupby('User_id').apply(
        lambda x: x.sort_values('review/time')['Title'].map(lambda title: map_title_to_int(title, title_to_int)).tolist()
    ).tolist()
    
    # 시퀀스 길이를 일정하게 맞추기 (예: 최대 50개 아이템)
    max_seq_length = 50
    padded_sequences = [seq[-max_seq_length:] + [0] * (max_seq_length - len(seq)) for seq in user_sequences]
    
    # 시퀀스를 텐서로 변환
    sequence_tensor = torch.LongTensor(padded_sequences)

    # 책 임베딩 텐서 생성
    book_embeddings = books_df['text_embedding'].apply(convert_to_list).tolist()
    book_embeddings = torch.tensor(book_embeddings, dtype=torch.float)

    # 패딩을 위한 특별한 임베딩 추가 (예: 모두 0인 벡터)
    padding_embedding = torch.zeros(book_embeddings.shape[1])
    book_embeddings = torch.cat([padding_embedding.unsqueeze(0), book_embeddings], dim=0)

    print(f"sequence_tensor shape: {sequence_tensor.shape}")
    print(f"book_embeddings shape: {book_embeddings.shape}")

    # DataLoader 생성
    dataset = BookSequenceDataset(sequence_tensor, book_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
