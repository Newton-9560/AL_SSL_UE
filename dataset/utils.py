from sentence_transformers import SentenceTransformer, util
import re, string
import numpy as np
import torch
import os

def fix_seed(seed=42):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cal_similarity(s1, s2):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    if s1 in s2 or s2 in s1:
        return 1.0

    embedding1 = model.encode(s1, convert_to_tensor=True)
    embedding2 = model.encode(s2, convert_to_tensor=True)

    similarity_score = util.cos_sim(embedding1, embedding2)

    return similarity_score.item()

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return s
