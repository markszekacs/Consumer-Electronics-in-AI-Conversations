# threshold_validation.py
# validates precision of the 0.15 similarity threshold on the keyword-matched corpus

from datasets import load_from_disk
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """Is this conversation about consumer electronics (laptops, phones, cameras, headphones, TVs, GPUs, smartwatches)?

Answer only YES or NO.

Conversation: {first_turn}"""


def llm_label(conversation):
    first_turn = conversation[0]['content'][:300]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(first_turn=first_turn)}],
        temperature=0,
        max_tokens=5
    )
    return 1 if response.choices[0].message.content.strip().upper() == "YES" else 0


if __name__ == '__main__':
    ds = load_from_disk('data/keyword_matched')
    similarities = np.load('data/similarities_keyword_matched.npy')

    # validate on all conversations above threshold
    mask = similarities >= 0.15
    indices = np.where(mask)[0]

    labels = []
    for i, idx in enumerate(indices):
        example = ds[int(idx)]
        label = llm_label(example['conversation'])
        labels.append({
            'index': int(idx),
            'similarity': similarities[idx],
            'label': label
        })
        if i % 100 == 0:
            print(f"Progress: {i}/{len(indices)}")

    df = pd.DataFrame(labels)
    df.to_csv('data/validation_labels.csv', index=False)

    precision = df['label'].mean()
    print(f"\nPrecision at threshold 0.15: {precision:.2%}")
    print(f"Relevant: {df['label'].sum()}/{len(df)}")