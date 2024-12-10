import torch
from sklearn.metrics import ndcg_score

def evaluate_model(model, test_data, books_df, k=10):
    model.eval()
    ndcg_scores = []
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            item_emb = torch.tensor(row['item_embedding']).unsqueeze(0)
            text_embs = torch.tensor(books_df['text_embedding'].tolist())
            
            item_hidden, _ = model.item_encoder(item_emb), model.text_encoder(text_embs)
            similarities = torch.cosine_similarity(item_hidden, text_embs)
            
            _, top_indices = torch.topk(similarities, k)
            recommended_items = books_df.iloc[top_indices]['asin'].tolist()
            
            true_item = row['asin']
            relevance = [1 if item == true_item else 0 for item in recommended_items]
            
            ndcg_scores.append(ndcg_score([relevance], [1]))
    
    print(f"NDCG@{k}: {sum(ndcg_scores)/len(ndcg_scores):.4f}")
