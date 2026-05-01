from config import *
from data_loader import load_data
from baseline_reranker import BaselineReranker
from query_routing import classify_query
from passage_compression import compress_passage
from utils import print_results

def main():
    queries, passages_data, _ = load_data()

    reranker = BaselineReranker(MODEL_NAME)

    for i in range(3):
        query = queries[i]
        passages = passages_data[i]["passage_text"]

        print(f"\n🔍 Query: {query}")

        # compress passages
        compressed = [compress_passage(query, p) for p in passages]

        # rerank
        results = reranker.rerank(query, compressed)

        # sort
        results = sorted(results, key=lambda x: x[1], reverse=True)

        print_results(results)

if __name__ == "__main__":
    main()
