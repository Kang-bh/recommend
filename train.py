import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
import time

# def train_stage1(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     criterion = nn.CrossEntropyLoss()

#     for batch in tqdm(train_loader, desc="Training"):
#         seq, labels = batch
#         seq, labels = seq.to(device), labels.to(device)

#         optimizer.zero_grad()
#         output = model(seq)
#         loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(train_loader)

def train_stage1(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in tqdm(train_loader, desc="Training"):
        seq, book_embeddings = batch
        seq, book_embeddings = seq.to(device), book_embeddings.to(device)

        print("Seq shape:", seq.shape)
        print("Book embeddings shape:", book_embeddings.shape)

        optimizer.zero_grad()
        output = model(seq)
        print("Model output shape:", output.shape)

        # 다음 아이템 예측을 위한 레이블 생성
        labels = seq[:, 1:].contiguous()
        output = output[:, :-1].contiguous()

        print("Reshaped output shape:", output.shape)
        print("Labels shape:", labels.shape)

        # 텐서 재구성
        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)

        print("Final output shape:", output.shape)
        print("Final labels shape:", labels.shape)

        # loss 계산
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def train_stage2(model, alignment_network, train_loader, optimizer, device):
    model.train()
    alignment_network.train()
    total_loss = 0
    criterion = nn.MSELoss()

    for batch in tqdm(train_loader, desc="Training"):
        seq, item_emb = batch
        seq, item_emb = seq.to(device), item_emb.to(device)

        optimizer.zero_grad()
        cf_emb = model(seq)
        # 마지막 시퀀스의 임베딩만 사용
        cf_emb = cf_emb[:, -1, :]
        aligned_emb = alignment_network(cf_emb)
        loss = criterion(aligned_emb, item_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
