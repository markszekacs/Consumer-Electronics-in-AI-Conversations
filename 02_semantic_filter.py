# 02_semantic_filter.py
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util
import numpy as np

# anchor is a collection of representative user queries rather than
# a meta-description - works better with sentence-transformers
ANCHOR = (
    "What is the best laptop to buy? Compare iPhone vs Samsung. "
    "Best GPU for gaming. OLED vs QLED TV differences. Cheapest place to buy a camera. "
    "How to return headphones to Amazon. Looking for a budget tablet under $300. "
    "My AirPods won't connect. DSLR camera recommendation for beginners. Best Airpods in 2023."
)

# chosen empirically - see threshold_validation.py for precision analysis
THRESHOLD = 0.15

def get_max_similarity(conversation, model, anchor_embedding):
    # max across turns: if any turn is relevant, the conversation counts
    user_turns = [t['content'][:512] for t in conversation if t['role'] == 'user']
    if not user_turns:
        return 0.0
    embeddings = model.encode(user_turns, convert_to_tensor=True)
    return util.cos_sim(embeddings, anchor_embedding).max().item()


if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    anchor_embedding = model.encode(ANCHOR, convert_to_tensor=True)

    for split in ['keyword_matched', 'non_keyword']:
        ds = load_from_disk(f'data/{split}')
        
        print(f"Computing similarities for {split} ({len(ds)} conversations)...")
        similarities = []
        for i, example in enumerate(ds):
            similarities.append(get_max_similarity(example['conversation'], model, anchor_embedding))
            if i % 1000 == 0:
                print(f"  {i}/{len(ds)}")
        
        similarities_np = np.array(similarities)
        np.save(f'data/similarities_{split}.npy', similarities_np)
        
        mask = similarities_np >= THRESHOLD
        filtered = ds.select(np.where(mask)[0].tolist())
        filtered.save_to_disk(f'data/semantic_filtered_{split}')
        
        print(f"  {split}: {len(ds)} -> {len(filtered)}")