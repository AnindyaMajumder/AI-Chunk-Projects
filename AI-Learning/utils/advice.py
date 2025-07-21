import os
import pandas as pd
from collections import defaultdict

def load_training_phrases_and_advices(csv_path):
    advices_by_category = defaultdict(list)
    if os.path.isdir(csv_path):
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(csv_path, filename), encoding="utf-8")
                for _, row in df.iterrows():
                    category = row.get('Category', row.get('category', 'General'))
                    advice = row.get('Advice', row.get('advice', ''))
                    if pd.notna(category) and pd.notna(advice):
                        advices_by_category[str(category).strip()].append(str(advice).strip())
    return advices_by_category

def format_advices_for_prompt(advices_by_category):
    lines = ["Reference Advice (imported from CSV):"]
    for category, advices in advices_by_category.items():
        lines.append(f"{category}:")
        for advice in advices:
            lines.append(f"  - {advice}")
    return "\n".join(lines)
