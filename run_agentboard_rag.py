# -*- coding: utf-8 -*-
# @Time    : 2024/06/27

import agentboard as ab
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def create_embedding(query_list, dim=64):
    """
        query_list: list of text
        dim: int
        You can choose vanilla word2vec/bert/openai embedding        
    """
    query_vec = np.random.rand(len(query_list), dim)
    return query_vec

def vector_search(query_vec, top_n, n_doc=200, dim = 64):
    """
        dummy faiss example
        
        input:
            query_vec: 2D Numpy Array
            top_n: int
            dim: int
        output: 
            indices: 2D list
            distances: 2D list
    """
    embeddings = np.random.rand(n_doc, dim)
    # embeddings = flattened_df['embeddings'].to_list()

    # Create the search index
    nbrs = NearestNeighbors(n_neighbors=top_n, algorithm='ball_tree').fit(embeddings)

    # To query the index, you can use the kneighbors method
    distances_array, indices_array = nbrs.kneighbors(query_vec)

    # Store the indices and distances in the DataFrame
    indices = indices_array.tolist()
    distances = distances_array.tolist()

    return indices, distances

def split_text(text, max_length, min_length):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) < max_length and len(' '.join(current_chunk)) > min_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # If the last chunk didn't reach the minimum length, add it anyway
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def load_paragraph():
    import pandas as pd
    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['path', 'text'])
    # splitting our data into chunks
    data_paths= ["./rag_docs.txt"]
    df_list = []
    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            file_content = file_content.replace("\n", "")
        # Append the file path and text to the DataFrame
        df_list.append(pd.DataFrame({'path': [path], 'text': [file_content]}))
    df = pd.concat(df_list)
    return df

def main():
    """
        # RAG Demo from Microsoft BLOG: 
        https://github.com/microsoft/generative-ai-for-beginners/blob/main/15-rag-and-vector-databases/notebook-rag-vector-databases.ipynb
    """
    ## RAG preprocess_docs
    docs_df = load_paragraph()
    chunk_list = []
    for text in docs_df['text']:
        chunk_list.extend(split_text(text, 400, 300))
    # RAG Result
    docs = []
    scores = []
    with ab.summary.FileWriter(logdir="./log") as writer:
        ## RAG Start: Query Embedding: Code for query vector search of your choice
        query_list = ["What are multi modal generative models?", "What's the difference between multi-modal generative models and large language models?"]
        query_vec = create_embedding(query_list)
        indices, distances = vector_search(query_vec, top_n=5, n_doc= len(chunk_list), dim=64)
        for group in indices:
            docs.append([{"doc_id":ind, "content":chunk_list[ind]} for ind in group])    
        for group in distances:
            scores.append([10.0 - d for d in group])

        ## agentboard logging start
        pipe = ab.summary.RAGPipeline(process_id="RAG", workflow_type="rag")
        pipe.input(query=query_list, embedding = query_vec.tolist())
        pipe.output(docs=docs, scores = scores,  key_doc_id="doc_id", key_content="content")
        pipe.write()

if __name__ == '__main__':
    main()
