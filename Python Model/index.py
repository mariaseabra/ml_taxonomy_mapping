import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from get_csv_file import parse_csv_file
from get_embeddings import get_embeddings_object
from compute_embeddings import compute_embeddings

def extract_link_name(input_str):
    import re
    match = re.search(r'=HYPERLINK\(".*?",\s*"(.*?)"\)', input_str)
    return match.group(1) if match else input_str

def find_best_match(parsed_embedding, own_embeddings, own_categories):
    similarities = cosine_similarity([parsed_embedding], own_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return {
        'bestMatch': own_categories[best_match_idx],
        'score': similarities[best_match_idx]
    }

def map_categories(parsed_objects, base_taxonomy_path):
    own_embeddings_complete = get_embeddings_object(base_taxonomy_path)
    own_embeddings = [obj['embeddings'] for obj in own_embeddings_complete]
    own_combined = [obj['combinedText'] for obj in own_embeddings_complete]

    ready_to_parse_objects = ['||'.join([obj['src_pt'], obj['src_cat'], obj['src_sc']]) for obj in parsed_objects]
    parsed_embedding_objects = compute_embeddings(ready_to_parse_objects)
    
    final_object = {}
    for parsed_combined, parsed_embedding in zip(ready_to_parse_objects, parsed_embedding_objects):
        best_match_mapped = find_best_match(parsed_embedding, own_embeddings, own_combined)
        final_object[parsed_combined] = best_match_mapped

    return final_object

def main():
    import sys
    from pathlib import Path

    args = sys.argv[1:]
    file_to_map_path = [arg.split('=')[1] for arg in args if arg.startswith('--file-to-map-path')][0]
    base_taxonomy_path = [arg.split('=')[1] for arg in args if arg.startswith('--base-taxonomy-path')][0]

    parsed_objects = parse_csv_file(file_to_map_path)
    for obj in parsed_objects:
        obj['src_cat'] = extract_link_name(obj['src_cat'])
        obj['src_pt'] = extract_link_name(obj['src_pt'])
        obj['src_sc'] = extract_link_name(obj['src_sc'])

    mapped_data = map_categories(parsed_objects, base_taxonomy_path)

    final_object = []
    for data in parsed_objects:
        key = f"{data['src_pt']}||{data['src_cat']}||{data['src_sc']}"
        founded_mapped = mapped_data[key]
        mapped_src_pt, mapped_src_cat, mapped_src_sc = founded_mapped['bestMatch'].split('||')
        final_object.append({
            **data,
            'ent_pt_2': mapped_src_pt,
            'ent_cat_2': mapped_src_cat,
            'ent_sc_2': mapped_src_sc,
            'score': founded_mapped['score']
        })

    output_path = Path('mapped_taxonomies') / f"{Path(file_to_map_path).stem}_ai_mapped.csv"
    pd.DataFrame(final_object).to_csv(output_path, index=False)
    print('CSV file successfully processed and created')

if __name__ == "__main__":
    main()
