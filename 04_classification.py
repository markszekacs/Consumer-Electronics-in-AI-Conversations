# 04_classification.py
from datasets import load_from_disk
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """You are analyzing a user conversation with an AI assistant about consumer electronics.

Classify this conversation by:

1. PRODUCT_CATEGORY (pick exactly one):
- Laptop/Computer
- Phone/Tablet
- Audio (headphones, earbuds, speakers)
- TV/Display (televisions, monitors, projectors)
- Camera/Photography (cameras, DSLR, mirrorless)
- GPU/Gaming (graphics cards, gaming PCs, peripherals)
- Other (smartwatches, smart home, accessories)

2. INTENTS (pick ALL that apply):
- Upper Funnel: user is exploring, learning what they want
- Mid Funnel: user is comparing specific options
- Lower Funnel: user is ready to buy, looking for deals/prices
- Post-purchase: user already bought, needs support/returns
- Informational: general knowledge, no purchase intent
- Generative: user wants content created (reviews, articles)
- Conversational: casual chat, opinions

Conversation:
{conversation_text}

Respond in JSON only, no other text:
{{"product_category": "...", "intents": ["...", "..."]}}"""


def classify(conversation):
    user_turns = [t['content'] for t in conversation if t['role'] == 'user']
    conversation_text = "\n".join([f"Turn {i+1}: {turn[:300]}" for i, turn in enumerate(user_turns)])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(conversation_text=conversation_text)}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


if __name__ == '__main__':
    ds = load_from_disk('data/final_dataset')
    
    results = []
    errors = []

    for i, example in enumerate(ds):
        try:
            result = classify(example['conversation'])
            result['conversation_id'] = example['conversation_id']
            result['timestamp'] = example['timestamp']
            results.append(result)
        except Exception as e:
            errors.append({'index': i, 'error': str(e)})

        if i % 50 == 0:
            print(f"Progress: {i}/{len(ds)}")

    print(f"Done. {len(results)} classified, {len(errors)} errors.")

    pd.DataFrame(results).to_csv('data/classifications.csv', index=False)