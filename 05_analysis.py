# 05_analysis.py
import pandas as pd
import ast
import numpy as np

def safe_literal_eval(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []


def load_data():
    df = pd.read_csv('data/classifications.csv')
    df['intents'] = df['intents'].apply(safe_literal_eval)
    df = df[df['intents'].map(len) > 0]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    
    # normalize inconsistent labels from LLM
    df['product_category'] = df['product_category'].replace({
        'TV/Display (televisions, monitors, projectors)': 'TV/Display',
        'Audio': 'Audio (headphones, earbuds, speakers)'
    })
    
    df['has_upper'] = df['intents'].apply(lambda x: 'Upper Funnel' in x)
    df['has_mid'] = df['intents'].apply(lambda x: 'Mid Funnel' in x)
    df['has_lower'] = df['intents'].apply(lambda x: 'Lower Funnel' in x)
    
    return df


def intent_distribution(df):
    df_exploded = df.explode('intents')
    df_exploded['intents'] = df_exploded['intents'].str.strip()
    # remove occasional LLM hallucination
    df_exploded = df_exploded[df_exploded['intents'] != 'Generational']
    return df_exploded


def print_overview(df, df_exploded):
    print("=== Product Category Distribution ===")
    print(df['product_category'].value_counts())
    
    print("\n=== Intent Distribution ===")
    print(df_exploded['intents'].value_counts())


def print_temporal(df_exploded):
    temporal = df_exploded.groupby(['month', 'intents']).size().unstack(fill_value=0)
    temporal_pct = temporal.div(temporal.sum(axis=1), axis=0) * 100
    print("\n=== Commercial Intent Trends ===")
    print(temporal_pct[['Upper Funnel', 'Mid Funnel', 'Lower Funnel']].round(1))


def print_category_breakdown(df_exploded):
    cross = df_exploded.groupby(['product_category', 'intents']).size().unstack(fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    print("\n=== Intent Distribution by Category ===")
    print(cross_pct[['Upper Funnel', 'Mid Funnel', 'Lower Funnel']].round(1))


def print_conversion_rates(df):
    print("\n=== Overall Funnel Conversion ===")
    upper = df['has_upper'].sum()
    mid = df['has_mid'].sum()
    print(f"Upper → Mid:   {(df['has_upper'] & df['has_mid']).sum() / upper * 100:.1f}%")
    print(f"Upper → Lower: {(df['has_upper'] & df['has_lower']).sum() / upper * 100:.1f}%")
    print(f"Mid → Lower:   {(df['has_mid'] & df['has_lower']).sum() / mid * 100:.1f}%")

    print("\n=== Conversion Rates by Category ===")
    for category in sorted(df['product_category'].unique()):
        cat = df[df['product_category'] == category]
        upper = cat['has_upper'].sum()
        mid = cat['has_mid'].sum()
        if upper < 5:
            continue
        u_m = (cat['has_upper'] & cat['has_mid']).sum() / upper * 100
        u_l = (cat['has_upper'] & cat['has_lower']).sum() / upper * 100
        m_l = (cat['has_mid'] & cat['has_lower']).sum() / mid * 100 if mid > 0 else 0
        print(f"{category} (n={len(cat)}): Upper→Mid {u_m:.1f}%, Upper→Lower {u_l:.1f}%, Mid→Lower {m_l:.1f}%")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df = load_data()
    df_exploded = intent_distribution(df)

    print_overview(df, df_exploded)
    print_temporal(df_exploded)
    print_category_breakdown(df_exploded)
    print_conversion_rates(df)