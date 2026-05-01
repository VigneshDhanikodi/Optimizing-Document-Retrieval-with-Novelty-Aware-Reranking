from datasets import load_dataset

def load_data(sample_size=500):
    dataset = load_dataset("ms_marco", "v1.1", split="validation")

    queries = dataset["query"][:sample_size]
    passages = dataset["passages"][:sample_size]
    labels = dataset["passages"][:sample_size]

    return queries, passages, labels
