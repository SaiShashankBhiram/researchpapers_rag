import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from attention_yolo.logger import logger
from attention_yolo.components.data_retreiver import query_rag
from attention_yolo.exception import CustomException
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Expanded ground truth answers
ground_truth_answers = {
    "What is attention mechanism?": "Attention mechanism helps models focus on important input parts.",
    "How does transformer architecture work?": "Transformers use self-attention for efficient processing.",
    "What is selective attention in psychology?": "Selective attention refers to the brain's ability to focus on relevant information while filtering out distractions."
}

# Function to calculate ROUGE score
def calculate_rouge(predicted, expected):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(expected, predicted)
    return {key: score.fmeasure for key, score in scores.items()}

# Function to evaluate query performance
def evaluate_query(question: str, expected_answer: str):
    try:
        predicted_response = query_rag(question, top_k=3)

        # Compute BLEU Score (adjusted for full sentence comparisons)
        bleu_score = sentence_bleu([expected_answer.split()], predicted_response.split(), weights=(0.5, 0.5))

        # Compute ROUGE Score
        rouge_scores = calculate_rouge(predicted_response, expected_answer)

        # Compute Cosine Similarity (for embedding closeness)
        try:
            predicted_embedding = np.array(query_rag(question, top_k=3)).reshape(1, -1)
            expected_embedding = np.array(query_rag(expected_answer, top_k=3)).reshape(1, -1)
            similarity_score = cosine_similarity(predicted_embedding, expected_embedding)[0][0]
        except Exception as e:
            similarity_score = None
            logger.warning(f"⚠️ Could not compute embedding similarity: {e}")

        logger.info(f"🔍 Query: {question}")
        logger.info(f"✅ Expected: {expected_answer}")
        logger.info(f"📝 Predicted: {predicted_response}")
        logger.info(f"📊 BLEU Score: {bleu_score}")
        logger.info(f"📊 ROUGE Scores: {rouge_scores}")
        logger.info(f"📊 Cosine Similarity Score: {similarity_score}")

        return {
            "question": question,
            "expected": expected_answer,
            "predicted": predicted_response,
            "bleu_score": bleu_score,
            "rouge_scores": rouge_scores,
            "similarity_score": similarity_score
        }

    except Exception as e:
        logger.error(f"❌ Error evaluating query: {e}")
        raise CustomException(e)

# Function to evaluate multiple queries
def batch_evaluation():
    results = []
    try:
        for question, expected_answer in ground_truth_answers.items():
            results.append(evaluate_query(question, expected_answer))

        logger.info(f"📈 Completed batch evaluation!")
        return results

    except Exception as e:
        logger.error(f"❌ Error in batch evaluation: {e}")
        raise CustomException(e)

if __name__ == "__main__":
    evaluation_results = batch_evaluation()
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info("📊 Evaluation results saved successfully!")