from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compress_passage(query, passage, top_k=3):
    sentences = passage.split(".")
    query_emb = model.encode(query, convert_to_tensor=True)

    scores = []
    for sent in sentences:
        emb = model.encode(sent, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, emb).item()
        scores.append((sent, sim))

    top_sentences = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return ". ".join([s[0] for s in top_sentences])
