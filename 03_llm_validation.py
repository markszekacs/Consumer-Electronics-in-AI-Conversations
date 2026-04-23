# 03_llm_validation.py
from datasets import load_from_disk, concatenate_datasets
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """You are filtering conversations for relevance to consumer electronics retail and purchasing.

Answer YES if the conversation is about:
- Buying, comparing, or researching consumer electronics (laptops, phones, tablets, cameras, headphones, TVs, GPUs, smartwatches)
- Technical support or returns for consumer electronics
- General information about consumer electronics products

Answer NO if the conversation is about:
- Coding, software development, or IT infrastructure
- Creative writing, essays, or content generation unrelated to electronics
- General chat, jokes, or unrelated topics
- Electronics only mentioned incidentally

First message: {first_turn}

Answer only YES or NO."""


def is_relevant(conversation):
    first_turn = conversation[0]['content'][:300]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(first_turn=first_turn)}],
        temperature=0,
        max_tokens=5
    )
    return response.choices[0].message.content.strip().upper() == "YES"


if __name__ == '__main__':
    validated = []

    for split in ['keyword_matched', 'non_keyword']:
        ds = load_from_disk(f'data/semantic_filtered_{split}')
        print(f"Validating {split} ({len(ds)} conversations)...")

        relevant_indices = []
        for i, example in enumerate(ds):
            try:
                if is_relevant(example['conversation']):
                    relevant_indices.append(i)
            except Exception:
                pass

            if i % 100 == 0:
                print(f"  {i}/{len(ds)}, relevant so far: {len(relevant_indices)}")

        clean = ds.select(relevant_indices)
        clean.save_to_disk(f'data/validated_{split}')
        print(f"  {split}: {len(ds)} -> {len(clean)}")
        validated.append(clean)

    # combine both pipelines into final dataset
    final = concatenate_datasets(validated)
    final.save_to_disk('data/final_dataset')
    print(f"Final dataset: {len(final)}")