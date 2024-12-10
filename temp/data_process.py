import pandas as pd
import numpy as np


def get_test_data(difficulty='light'):
    df_reviews = pd.read_csv('reviews.csv')

    if difficulty == 'light':

        train_data = df[:3600]
        valid_data = df[3600:4800]
        test_data = df[4800:6000]                
      
        # 데이터셋 크기 확인
        print(f"훈련 데이터: {len(train_data)}개")
        print(f"검증 데이터: {len(valid_data)}개")
        print(f"테스트 데이터: {len(test_data)}개")
    elif difficulty == "origin":
        train_val_df, test_data = train_test_split(df_reviews, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_val_df, test_size=0.2, random_state=42)


        print(f"훈련 데이터: {len(train_data)}개")
        print(f"검증 데이터: {len(valid_data)}개")
        print(f"테스트 데이터: {len(test_data)}개")