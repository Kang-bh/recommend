
# Precision, Recall, F1-Score 평가
def evaluate_precision_recall_f1(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# DCG 계산
def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
    return 0.0

# NDCG 평가
def ndcg_at_k(true_relevance, predicted_ranks, k=5):
    ideal_relevance = sorted(true_relevance, reverse=True)
    dcg = dcg_at_k([true_relevance[i] for i in predicted_ranks[:k]], k)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0

# Average Precision 계산
def average_precision_at_k(true_relevance, predicted_ranks, k=5):
    correct_predictions = 0
    score = 0.0
    for i, idx in enumerate(predicted_ranks[:k]):
        if true_relevance[idx] == 1:
            correct_predictions += 1
            score += correct_predictions / (i + 1)
    return score / min(k, sum(true_relevance))

# MAP 계산
def mean_average_precision(true_relevances, predicted_ranks_list, k=5):
    return np.mean([average_precision_at_k(true_relevance, predicted_ranks, k)
                    for true_relevance, predicted_ranks in zip(true_relevances, predicted_ranks_list)])
