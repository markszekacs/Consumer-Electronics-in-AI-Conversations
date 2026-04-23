from datasets import load_dataset
import re

KEYWORDS = [
    'laptop', 'notebook', 'macbook', 'thinkpad',
    'smartphone', 'iphone', 'android', 'samsung', 'pixel',
    'headphones', 'earbuds', 'airpods',
    'tablet', 'ipad',
    'gpu', 'graphics card', 'nvidia', 'amd',
    'tv', 'television', 'monitor',
    'camera', 'dslr', 'mirrorless',
    'smartwatch', 'apple watch',
    'speaker', 'soundbar'
]

pattern = re.compile('|'.join(KEYWORDS), re.IGNORECASE)

def is_relevant(example):
    if example['language'] != 'English' or example['toxic']:
        return False
    return bool(pattern.search(example['conversation'][0]['content']))

def is_non_keyword(example):
    if example['language'] != 'English' or example['toxic']:
        return False
    return not bool(pattern.search(example['conversation'][0]['content']))


if __name__ == '__main__':
    ds = load_dataset("allenai/WildChat")
    
    keyword_matched = ds['train'].filter(is_relevant)
    keyword_matched.save_to_disk('data/keyword_matched')
    print(f"Keyword matched: {len(keyword_matched)}")
    
    non_keyword = ds['train'].filter(is_non_keyword)
    non_keyword.save_to_disk('data/non_keyword')
    print(f"Non-keyword: {len(non_keyword)}")