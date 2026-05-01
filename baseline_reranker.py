from sentence_transformers import CrossEncoder

class BaselineReranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, passages):
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        return list(zip(passages, scores))
