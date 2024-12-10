import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Azure OpenAI 설정
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

# 모델 설정
ITEM_DIM = 768
TEXT_DIM = 768
HIDDEN_DIM = 512
LLM_DIM = 768

# 학습 설정
BATCH_SIZE1 = 128
BATCH_SIZE2 = 128
EPOCHS = 10
LEARNING_RATE1 = 0.0001
LEARNING_RATE2 = 0.0001

# 데이터 경로
BOOKS_PATH = os.path.join(current_file_dir, '..', 'book', 'book_metadata.csv')
REVIEWS_PATH = os.path.join(current_file_dir, '..', 'book', 'reviews.csv')

# Azure OpenAI 설정이 제대로 로드되었는지 확인
# if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
#     raise ValueError("Azure OpenAI 설정이 .env 파일에 올바르게 설정되지 않았습니다.")

MAX_SEQ_LENGTH = 50
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2
