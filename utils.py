def print_results(results):
    for i, (text, score) in enumerate(results[:5]):
        print(f"{i+1}. Score: {score:.4f}")
        print(text[:100])
        print("-" * 40)
