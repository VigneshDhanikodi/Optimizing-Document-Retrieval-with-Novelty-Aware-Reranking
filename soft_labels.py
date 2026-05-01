from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_soft_label(passage, answer):
    score = scorer.score(passage, answer)
    return score['rougeL'].fmeasure
