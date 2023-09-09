import numpy as np

def precision(q):
    if not isinstance(q, list):
        raise ValueError("q debe ser una lista")
    
    if all(item in [0, 1] for item in q):
        return sum(q) / len(q)
    else:
        raise ValueError("q debe contener solo ceros (0) y unos (1)")
    

def precision_at_k(q, k):
    if not isinstance(q, list):
        raise ValueError("q debe ser una lista")
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo")
    
    if all(item in [0, 1] for item in q):
        return sum(q[:k]) / k
    else:
        raise ValueError("q debe contener solo ceros (0) y unos (1)")

def recall_at_k(q, k, number_relevant_docs):
    if not isinstance(q, list):
        raise ValueError("q debe ser una lista")
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo")
    
    if not isinstance(number_relevant_docs, int) or number_relevant_docs < 0:
        raise ValueError("number_relevant_docs debe ser un entero no negativo")
    
    if all(item in [0, 1] for item in q):
        relevant_retrieved = sum(q[:k])
        return relevant_retrieved / number_relevant_docs
    else:
        raise ValueError("q debe contener solo ceros (0) y unos (1)")
    
def average_precision(q):
    if not isinstance(q, list):
        raise ValueError("q debe ser una lista")
    
    if all(item in [0, 1] for item in q):
        total_relevant_docs = sum(q)
        accumulated_precision = 0.0
        relevant_count = 0
        
        if total_relevant_docs == 0:
            return 0.0

        for i, relevance in enumerate(q):
            if relevance == 1:
                relevant_count += 1
                accumulated_precision += precision_at_k(q, i+1)
        
        return accumulated_precision / total_relevant_docs
    else:
        raise ValueError("q debe contener solo ceros (0) y unos (1)")

def mean_average_precision(query_results):
    total_queries = len(query_results)
    sum_average_precision = 0.0
    
    for q in query_results:
        sum_average_precision += average_precision(q)
    
    return sum_average_precision / total_queries

def dcg_at_k(relevance_vector, k):
    if not isinstance(relevance_vector, list):
        raise ValueError("relevance_vector debe ser una lista")
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo")
    
    dcg_value = relevance_vector[0]
    
    for i in range(1, min(k, len(relevance_vector))):
        dcg_value += relevance_vector[i] / (np.log2(i + 1))
    
    return dcg_value

def ndcg_at_k(relevance_vector, k):
    if not isinstance(relevance_vector, list):
        raise ValueError("relevance_vector debe ser una lista")
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo")
    
    dcg_value = dcg_at_k(relevance_vector, k)
    ideal_dcg_value = dcg_at_k(sorted(relevance_vector, reverse=True), k)
    
    if ideal_dcg_value == 0:
        return 0.0
    
    ndcg_value = dcg_value / ideal_dcg_value
    return ndcg_value