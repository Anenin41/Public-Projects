# This Python script reads the 10k business documents, searches for appropriate #
# keywords regarding digitalization and returns a measure for each 10 filling.  #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Fri 25 Apr 2025 @ 10:48:44 +0200
# Modified: Fri 25 Apr 2025 @ 19:16:01 +0200

# Packages
import pandas as pd
import os
import re
from collections import defaultdict
from pathlib import Path

# Read the keyword dictionary
df_dict = pd.read_excel("full_dictionary.xlsx")
category_keywords = {}

# Convert each column into a different category, i.e. a list of keywords map.
for col in df_dict.columns:
    words = df_dict[col].dropna().str.lower().tolist()
    # Replace "-" for "_" for all words in the dictionary.
    category_keywords[col] = [w.replace("-", "_") for w in words]

# Normalize text files, i.e. convert whitespaces and hyphens to underscores.
root_folder = Path("Business")
text_files = list(root_folder.rglob("*.txt"))

print(f"Found {len(text_files)} text files to process.")

output = []

for filepath in text_files:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
            lowercase_text = raw_text.lower()
            # Spaces & hyphens to underscore
            normalized_text = re.sub(r"[\s\-]+", "_", lowercase_text)
        
            # total word count in the document
            word_count = len(re.findall(r"\b\w+\b", raw_text))
            total_hits = 0
            category_hits = {}

        for category, keywords in category_keywords.items():
            matches = [kw for kw in keywords if kw in normalized_text]
            category_hits[category] = matches
            total_hits += len(matches)

        # Save final values
        ## Clean relative path
        row = {"Filename": filepath.name}
        for category in category_keywords:
            row[category] = len(category_hits.get(category, []))
        row["Total_Matches"] = total_hits
        row["Word_Count"] = word_count
        row["Match_Ratio"] = round(total_hits / word_count, 4) if word_count > 0 else 0
        output.append(row)
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

df_out = pd.DataFrame(output)
df_out.to_excel("cluster_keyword_hits.xlsx", index=False)

print("Output saved to 'cluster_keyword_hits.xlsx'")
