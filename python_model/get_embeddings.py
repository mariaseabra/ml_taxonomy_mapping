from compute_embeddings import compute_embeddings

def get_embeddings_object(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    combined_texts = [line.split('||') for line in lines]  # Adjust based on your data format
    embeddings = compute_embeddings([' '.join(text) for text in combined_texts])
    return [{'combinedText': '||'.join(text), 'embeddings': emb} for text, emb in zip(combined_texts, embeddings)]
