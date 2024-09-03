import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def compute_embeddings(texts, vector_size=100, window=5, min_count=1):
    """
    Computes embeddings using Word2Vec for a list of texts.
    """
    # Train Word2Vec model on the input texts
    model = Word2Vec(sentences=[text.split() for text in texts], vector_size=vector_size, window=window, min_count=min_count, workers=4)
    
    # Compute embeddings by averaging word vectors for each text
    embeddings = []
    for text in texts:
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))  # Handle texts with no known words
    
    return embeddings

def generate_base_taxonomy_embeddings(input_file_path, output_file_path):
    """
    Generates embeddings for the base taxonomy and saves them to a CSV file.
    """
    # Load base taxonomy data
    base_taxonomy = pd.read_csv(input_file_path)
    
    # Combine the three columns into a single string for each row
    combined_texts = base_taxonomy.apply(lambda row: f"{row['productType']}||{row['category']}||{row['subCategory']}", axis=1).tolist()
    
    # Compute embeddings for each combined text
    embeddings = compute_embeddings(combined_texts)
    
    # Save the combined data with embeddings to a new CSV file
    base_taxonomy['combinedText'] = combined_texts
    base_taxonomy['embeddings'] = embeddings
    base_taxonomy.to_csv(output_file_path, index=False)
    print(f'Base taxonomy embeddings saved to {output_file_path}')

# Generate the embeddings for the base taxonomy
generate_base_taxonomy_embeddings('python_model/base_taxonomy/Taxonomy.csv', 'base_taxonomy_embeddings.csv')
