#!/usr/bin/env python

import re, os,sys, requests
from bs4 import BeautifulSoup
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document



###############################################################################
# Configuration / Constants
###############################################################################

# Confluence API info
CONFLUENCE_BASE_URL=os.getenv('AI_CONFLUENCE', 'https://wiki.test.net') + '/rest/api'
CONFLUENCE_API_TOKEN=os.getenv('CONFLUENCE_API_TOKEN', '')

# Directory where Chroma will store the vector index
CHROMA_PERSIST_DIR = "./db/confluence_chroma_db"

# Model Embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

# all-MiniLM-L6-v2 model can handle up to 256 tokens per chunk.
# It's good practice to use smaller chunks (e.g., 100-256 tokens) for better granularity in retrieval.
#VECTOR_CHUNK_SIZE=236      ## 256 - 20 (overlap)
VECTOR_CHUNK_SIZE=1000      ## 256 - 20 (overlap)
PAGE_SOURCE="VectorDB"


###########################################################
# Get all Pages that were modified since XXXX , per Search Query
###########################################################
def find_confluence_pages_per_lastmodfieddate(modifiedDate: str):

        headers = {
            "Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"
        }
        response = requests.get(f"{CONFLUENCE_BASE_URL}/content/search?cql=(type=page and space=ADH AND lastmodified >= {modifiedDate})",
                                 headers=headers)
        #print(response.json().get("results"))

        pagesList = response.json().get("results")

        return pagesList

###########################################################
# Get all Pages per Search Query
###########################################################
def find_confluence_pages():

        headers = {
            "Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"
        }
        response = requests.get(f"{CONFLUENCE_BASE_URL}/content/search?cql=(type=page and space=ADH) OR (creator=admin and type=blogpost)&expand=body.storage&limit=2",
                                 headers=headers)
        #print(response.json().get("results"))

        pagesList = response.json().get("results")

        return pagesList

##################################################################
# Get Page Details per PageID
##################################################################
def get_page_details(page_id):
    headers = {
        "Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"
    }

    #print(f"Incoming pageId to get content: {page_id}")
    response = requests.get(f"{CONFLUENCE_BASE_URL}/content/{page_id}?expand=body.storage",
                             headers=headers)
    return response.json()


##################################################################
# Retrieves direct child pages of a specified parent page by ID.
# Returns a list of child page objects.
##################################################################
def get_child_pages(parent_page_id):

    child_url = f"{CONFLUENCE_BASE_URL}/content/{parent_page_id}/child/page"
    headers = {
            "Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"
        }

    response = requests.get(
        child_url,
        headers=headers
    )
    
    response.raise_for_status()
    data = response.json()
    #print(f"Response: {data}")
    return data.get("results", [])

##################################################################
# Recursively fetches all descendants of a page, printing them hierarchically.
# Returns a list of all descendant page objects (for demonstration).
##################################################################
def get_all_descendants(page_id, depth=0):

    descendants = []

    # Get the immediate children of the current page
    child_pages = get_child_pages(page_id)

    for child in child_pages:
        descendants.append(child)

        # Print the child page title with some indentation
        #print("  " * depth + f"├─ {child['title']} (ID: {child['id']})")

        # Recursively get that child's descendants
        child_descendants = get_all_descendants(child["id"], depth + 1)
        descendants.extend(child_descendants)
    
    return descendants


##################################################################
#    Converts HTML content into a clean, structured text format for embeddings.
#    - Preserves headings, lists, and tables.
#    - Converts headings to Markdown style.
#    - Formats lists properly.
#    - Converts tables into Markdown format.
##################################################################
def html_to_structured_text(html_content: str) -> str:

    soup = BeautifulSoup(html_content, "html.parser")

    # Convert headings to Markdown-style
    for level in range(1, 7):
        for tag in soup.find_all(f'h{level}'):
            tag.replace_with(f"{'#' * level} {tag.get_text(strip=True)}\n\n")

    # Convert unordered lists to Markdown-style bullet points
    for ul in soup.find_all("ul"):
        text = "\n".join(f"- {li.get_text(strip=True)}" for li in ul.find_all("li"))
        ul.replace_with(text + "\n\n")

    # Convert ordered lists to Markdown-style numbered lists
    for ol in soup.find_all("ol"):
        text = "\n".join(f"{idx+1}. {li.get_text(strip=True)}" for idx, li in enumerate(ol.find_all("li")))
        ol.replace_with(text + "\n\n")

    # Convert tables into Markdown format
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["th", "td"])]
            rows.append(" | ".join(cols))

        if rows:
            header = rows[0]  # Assuming first row is the header
            separator = " | ".join(["---"] * len(rows[0].split(" | ")))  # Markdown separator
            markdown_table = f"{header}\n{separator}\n" + "\n".join(rows[1:]) if len(rows) > 1 else header
            table.replace_with(markdown_table + "\n\n")

    # Extract clean text
    text_content = soup.get_text("\n", strip=True)

    # Normalize excessive newlines
    text_content = re.sub(r'\n\s*\n+', '\n\n', text_content).strip()

    return text_content


##################################################################
# Break text into chunks for better search/retrieval
##################################################################
def chunk_text_old(text: str, max_tokens: int ) -> list:

    # Split the text into chunks for better vector search performance.
    # This is a simple approach that splits by word count. Adjust as needed.
    if not text:
        return []

    # Split the text into tokens (here, we simply use whitespace splitting)
    tokens = text.split()
    
    # Build chunks from tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        # Create a chunk from max_tokens tokens, join them back into a string
        chunk = " ".join(tokens[i:i + max_tokens])
        chunks.append(chunk)
    
    return chunks

##################################################################
# Break text into chunks for better search/retrieval
##################################################################
def chunk_text(text: str, max_tokens: int ) -> list:

    chunk_size = max_tokens  # Optimal for RAG
    #chunk_overlap = 20  # Helps retain context across chunks
    chunk_overlap = 100  # Helps retain context across chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_text(text)  # Split document into chunks
    return chunks


##################################################################
# UPSERT (Add/Update) data in vector store
##################################################################
def add_or_update_data(vector_store: VectorStore, pageDetails):
        body_value = pageDetails["body"]["storage"]["value"]
        clean_html_body = html_to_structured_text(body_value)
        #print("**********************************************************")
        #print(clean_html_body)
        #print("**********************************************************")
        
        print("Searching for page: " + pageDetails['id'])
        result = vector_store.get(where={"pageid": pageDetails['id']})
        if result.get('ids'):
            print(f"Deleting all pages related to {pageDetails['id']}")
            vector_store.delete(ids=result.get('ids'))  # remove by ID

        #print(f"Length of text: {len(clean_html_body)}")

        ### Append Title to content
        clean_html_body = pageDetails['title'] + "\n\n" + clean_html_body

        #chunksList = chunk_text(clean_html_body, VECTOR_CHUNK_SIZE)
        chunksList = chunk_text(clean_html_body, VECTOR_CHUNK_SIZE)
        print(f"# of chunks: {len(chunksList)} to store for this document")
        for i,chunk in enumerate(chunksList):
            #print("---------------------------------------------------------------")
            #print(f"Loading chunks for page: {pageDetails['title']}")
            #print(f"Chunk: {chunk}")

            doc_id = f"{pageDetails['id']}-{i}"
        
            document = Document(
                page_content=chunk,
                metadata={"title": pageDetails['title'], "source": PAGE_SOURCE, "pageid": pageDetails['id']}
            )

            vector_store.add_documents(
                documents=[document],
                ids=[doc_id]    
            )
        print("Following page added/updated in DB: " + pageDetails['id'])
            

 
###############################################################################
# Main logic 
# Sample command:  python addUpdatePagesVectorDB.py date=`date +"%Y-%m-%d"`  or  python test.py pageid=1234
#
# Sample PageIds for Testing (Dev):
#   pageid="219418745"   # Data Hub Knowledge Space
#   pageid="219417517"
###############################################################################
def main():
    searchPagesList = []
    
    print("\nStarting...")

    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    sys.stdout.reconfigure(encoding='utf-8')

    print("\nInitializing embedding model...")
    #embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,model_kwargs={"trust_remote_code": True, "device": "cpu"})  # Use "cuda" if using GPU
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
   

    print("\nCreating (or updating) Chroma vector store...")
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedder
        #model_kwargs={"device": "cpu"}  # Use "cpu" if no GPU available or "cuda" for GPU acceleration
    )
    
    ### Set up input parm patterns 
    date_pattern = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
    pageid_pattern = re.compile(r"^pageid=(\d+)$")
    
    if not sys.argv[1:]:
        print(f"Missing parameter: date=YYYY-MM-DD  or pageid=1234")
        sys.exit(1)
        
    print(f"Processing input parameter: {sys.argv[1:]}")    
    for arg in sys.argv[1:]:
        date_match = date_pattern.match(arg)
        pageid_match = pageid_pattern.match(arg)
        
        if date_match:
            print(f"Date has been provided to filter and grab only modified pages >= {date_match.group(1)}")
            searchPagesList = find_confluence_pages_per_lastmodfieddate(date_match.group(1))
        elif pageid_match:
            print(f"PageId has been provided - Grab all child pages and primary page for pageId: {pageid_match.group(1)}")
            searchPagesList = get_all_descendants(pageid_match.group(1))
        else:
            print(f"Unknown parameter - exiting...")
            sys.exit(1)
    
    for page in searchPagesList:
        print(f"----------------------------------------------------------")
        print(f"Processing Page: {page['id']} | {page['title']}")
        pageDetails = get_page_details(page["id"])
        if pageDetails:
            add_or_update_data(vector_store, pageDetails)
        print(f"----------------------------------------------------------")
                 
    ### Include the primary/top page details too (for processing pageid=XXXX)
    if pageid_match:
        pageDetails = get_page_details(pageid_match.group(1))
        if pageDetails:
            print(f"Parent/Top Page: {pageDetails['id']} | {pageDetails['title']}")
            add_or_update_data(vector_store, pageDetails)
 
   
    print(f"\nChroma vector database: {CHROMA_PERSIST_DIR} - {len(searchPagesList)} pages added/updated: ")
    

###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    main()

