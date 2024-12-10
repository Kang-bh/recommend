import numpy as np

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


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

# Average Precision 계산
def average_precision_at_k(true_relevance, predicted_ranks, k=5):
    correct_predictions = 0
    score = 0.0
    for i, idx in enumerate(predicted_ranks[:k]):
        if true_relevance[idx] == 1:
            correct_predictions += 1
            score += correct_predictions / (i + 1)
    return score / min(k, sum(true_relevance))

def hit_ratio(recommended_items, actual_items):
    return len(set(recommended_items) & set(actual_items)) / len(actual_items)


# MAP 계산
def mean_average_precision(true_relevances, predicted_ranks_list, k=5):
    return np.mean([average_precision_at_k(true_relevance, predicted_ranks, k)
                    for true_relevance, predicted_ranks in zip(true_relevances, predicted_ranks_list)])
