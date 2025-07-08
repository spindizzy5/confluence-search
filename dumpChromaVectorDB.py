#!/usr/bin/env python

import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def dump_chroma_db(persist_directory: str, dump_file: str):
    ##################################################################
    # Dump all data (documents, metadata, IDs, embeddings) from a Chroma vector DB into a JSON file.
    # :param persist_directory: Path to the directory where the Chroma DB is stored.
    # :param dump_file: Name of the file (JSON) to dump the data to.
    ##################################################################
    # Initialize the Chroma object, pointing to an existing vector store

    embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME
            )
    
    chroma_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    
    # Access the underlying collection
    collection = chroma_db._collection
    
    # Retrieve all data from the collection
    # Include the documents, metadata, embeddings, and IDs
    print("Getting data....")
    all_data = collection.get(include=["metadatas","documents"])
    
    print("Dumping data....")
    # Dump the retrieved data to a JSON file
    #with open(dump_file, "w", encoding="utf-8") as f:
    #    json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    #print(f"Chroma DB data has been dumped to '{dump_file}'.")
    json_string = json.dumps(all_data, indent=4)
    print(json_string)

if __name__ == "__main__":
    # Example usage:
    PERSIST_DIRECTORY = "./db/confluence_chroma_db"
    DUMP_FILE = "chroma_db_dump.json"
    
    dump_chroma_db(PERSIST_DIRECTORY, DUMP_FILE)
