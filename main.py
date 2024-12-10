import argparse
import torch
from models import SASRec, AlignmentNetwork
from train import train_stage1, train_stage2
from data_loader import load_data, create_dataloader
from config import *
from openai import AzureOpenAI
from transformers import pipeline
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def llm_inference(prompt):
#     client = AzureOpenAI(
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_key=AZURE_OPENAI_API_KEY,
#         api_version="2024-02-15-preview"
#     )
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful book recommendation assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content

def generate_llm_prompt(user_history, aligned_emb, books_df):
    prompt = f"User history: {user_history}\n"
    prompt += f"Aligned embedding: {aligned_emb.tolist()}\n"
    prompt += "Based on this information, recommend a book and explain why."
    return prompt

def llm_inference(prompt):
    generator = pipeline('text-generation', model='facebook/opt-1.3b')
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    total_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_load_start = time.time()
    books_df, train_data, test_data = load_data(BOOKS_PATH, REVIEWS_PATH)
    data_load_end = time.time()
    print(f"Data loading time: {data_load_end - data_load_start:.2f} seconds")

    if args.pretrain_stage1:
        stage1_start = time.time()
        model = SASRec(num_items=len(books_df), d_model=ITEM_DIM, nhead=NUM_HEADS, 
                       num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE1)
        train_loader = create_dataloader(train_data, books_df, BATCH_SIZE1)

        for epoch in range(EPOCHS):
            epoch_start = time.time()
            loss = train_stage1(model, train_loader, optimizer, device)
            epoch_end = time.time()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Time: {epoch_end - epoch_start:.2f} seconds")

        torch.save(model.state_dict(), 'sasrec_model.pth')
        stage1_end = time.time()
        print(f"Stage 1 training time: {stage1_end - stage1_start:.2f} seconds")

    elif args.pretrain_stage2:
        stage2_start = time.time()
        print("Starting Stage 2 pre-training...")

        print("Initializing SASRec model...")
        model = SASRec(num_items=len(books_df), d_model=ITEM_DIM, nhead=NUM_HEADS, 
                    num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        print("SASRec model initialized.")

        print("Loading SASRec model weights...")
        model.load_state_dict(torch.load('sasrec_model.pth'))
        print("SASRec model weights loaded successfully.")

        print("Initializing Alignment Network...")
        alignment_network = AlignmentNetwork(ITEM_DIM, LLM_DIM).to(device)
        print("Alignment Network initialized.")

        print("Setting up optimizer...")
        optimizer = torch.optim.Adam(list(model.parameters()) + list(alignment_network.parameters()), lr=LEARNING_RATE2)
        print("Optimizer set up complete.")

        print("Creating data loader...")
        train_loader = create_dataloader(train_data, books_df, BATCH_SIZE2)
        print("Data loader created.")

        print("Starting training loop...")
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS} starting...")
            epoch_start = time.time()
            loss = train_stage2(model, alignment_network, train_loader, optimizer, device)
            epoch_end = time.time()
            print(f"Epoch {epoch+1}/{EPOCHS} completed.")
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Time: {epoch_end - epoch_start:.2f} seconds")

        print("Training loop completed.")

        print("Saving Alignment Network...")
        torch.save(alignment_network.state_dict(), 'alignment_network.pth')
        print("Alignment Network saved successfully.")

        stage2_end = time.time()
        print(f"Stage 2 training completed. Total time: {stage2_end - stage2_start:.2f} seconds")


    elif args.inference:
        inference_start = time.time()
        model = SASRec(num_items=len(books_df), d_model=ITEM_DIM, nhead=NUM_HEADS, 
                       num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        model.load_state_dict(torch.load('sasrec_model.pth'))
        
        alignment_network = AlignmentNetwork(ITEM_DIM, LLM_DIM).to(device)
        alignment_network.load_state_dict(torch.load('alignment_network.pth'))

        user_history = test_data[test_data['reviewerID'] == test_data['reviewerID'].iloc[0]]
        user_sequence = user_history['asin'].tolist()

        embedding_start = time.time()
        with torch.no_grad():
            cf_emb = model(torch.tensor(user_sequence).unsqueeze(0).to(device))
            aligned_emb = alignment_network(cf_emb[:, -1, :])
        embedding_end = time.time()
        print(f"Embedding generation time: {embedding_end - embedding_start:.2f} seconds")

        llm_start = time.time()
        prompt = generate_llm_prompt(user_sequence, aligned_emb.cpu(), books_df)
        recommendation = llm_inference(prompt)
        llm_end = time.time()
        print(f"LLM inference time: {llm_end - llm_start:.2f} seconds")

        print("User History:", user_sequence)
        print("LLM Recommendation:", recommendation)
        inference_end = time.time()
        print(f"Total inference time: {inference_end - inference_start:.2f} seconds")

    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
