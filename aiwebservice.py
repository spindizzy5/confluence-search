#!/usr/bin/env python

###############################################
# To start up web service, run following:
# uvicorn aiwebservice:app --reload
###############################################
import json,os,sys,re,requests, time, unicodedata
from typing import Optional
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from functools import wraps
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy  # Load spaCy English model


### Initilization
AI_SERVER = os.getenv('AI_SERVER', 'http://localhost')
AI_CONFLUENCE = os.getenv('AI_CONFLUENCE', 'https://wiki.test.net')
AI_OLLAMA_SERVER = os.getenv('AI_OLLAMA_SERVER', 'localhost')
AI_OLLAMA_PORT = os.getenv('AI_OLLAMA_PORT', '11434')
AI_SERVICE_PORT = os.getenv('AI_SERVICE_PORT', '8080')
AI_UI_PORT = os.getenv('AI_UI_PORT', '3000')
AI_MODEL = os.getenv('AI_MODEL', 'llama3.1:8b')


ENABLE_CHAT_CONVERSATION=False

# Confluence API info
CONFLUENCE_REST_URL=AI_CONFLUENCE + '/rest/api'
CONFLUENCE_API_TOKEN=os.getenv('CONFLUENCE_API_TOKEN', '')
#--------------------------------
MAX_NUM_CHUNKS_RETURN: int = 10    ### MAX Num of documents to return from Vector search for ranking/filtering
MIN_SCORE_THRESHOLD = 0.80
MAX_SCORE_THRESHOLD = 1.30
MIN_PAGE_CONTENT_SIZE_CHARS = 100  ### Page content must have minimum number of characters to be accepted
#--------------------------------
MAX_INFER_PAGES=5
MAX_HIGH_RELEVANT_PAGES=3
MAX_LOW_RELEVENT_PAGES=2
MAX_AI_RELEVANT_PAGES=3
MAX_SEARCH_PAGES_INFER=4
#--------------------------------
MAX_LINKS_VECTORDB=3   # For displaying links from search results  
MAX_LINKS_SEARCHAPI=3  # For displaying links from search results
#--------------------------------
MAX_SEARCH_QUERY_LENGTH=180
SEARCH_SOURCE_PAGE="SearchAPI"
TOKEN_NUM_REDUCER=0.75  # Percentage to reduce to closely match num using Model/Tokenizer


# Initialize FastAPI
app = FastAPI()

UI_SERVER=AI_SERVER+":"+AI_UI_PORT

app.add_middleware(
    CORSMiddleware,
    allow_origins=[UI_SERVER],  # or ["*"] for testing
    #allow_origins=["*"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# CONFIGURATION
###############################################################################
config_chroma_persist_dir: str = "./db/confluence_chroma_db"
config_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
config_ollama_model_name: str = AI_MODEL
config_ollama_server_url: str = AI_OLLAMA_SERVER + ":" + AI_OLLAMA_PORT
config_temperature: float = 0.7
config_max_tokens: int = 2000


# Global objects (initialize once at startup)
embedding_model = None
vectorstore = None
chat_histories = {}

###################################################
# ONE-TIME STARTUP / INITIALIZATION
###################################################
try:
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    sys.stdout.reconfigure(encoding='utf-8')
    nlp = spacy.load("en_core_web_sm")

    print("Loading HuggingFace embedding model...")
    embedding_model = HuggingFaceEmbeddings(
       model_name=config_embedding_model_name,
    )
            
    print(f"Loading Chroma from directory: {config_chroma_persist_dir}")
    vectorstore = Chroma(
        persist_directory=config_chroma_persist_dir,
        embedding_function=config_embedding_model_name,
    )

    print(f"Loading configurations...")

except Exception as e:
    print(f"Initialization failed: {str(e)}")


##############################################################
# Truncate Query for searching 
##############################################################
def load_ai_config(file_path="aiconfig.txt"):
    global MIN_SCORE_THRESHOLD, MAX_SCORE_THRESHOLD, MIN_PAGE_CONTENT_SIZE_CHARS
    global MAX_INFER_PAGES, MAX_HIGH_RELEVANT_PAGES, MAX_LOW_RELEVENT_PAGES
    global MAX_AI_RELEVANT_PAGES, MAX_SEARCH_PAGES_INFER, MAX_LINKS_VECTORDB
    global MAX_LINKS_SEARCHAPI

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")
    
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                try:
                    # Convert numeric values appropriately
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {key}: {value}")
                
    # Assign global variables
    MIN_SCORE_THRESHOLD = config.get("MIN_SCORE_THRESHOLD", 0.80)
    MAX_SCORE_THRESHOLD = config.get("MAX_SCORE_THRESHOLD", 1.30)
    MIN_PAGE_CONTENT_SIZE_CHARS = config.get("MIN_PAGE_CONTENT_SIZE_CHARS", 100)
    MAX_INFER_PAGES = config.get("MAX_INFER_PAGES", 5)
    MAX_HIGH_RELEVANT_PAGES = config.get("MAX_HIGH_RELEVANT_PAGES", 3)
    MAX_LOW_RELEVENT_PAGES = config.get("MAX_LOW_RELEVENT_PAGES", 2)
    MAX_AI_RELEVANT_PAGES = config.get("MAX_AI_RELEVANT_PAGES", 3)
    MAX_SEARCH_PAGES_INFER = config.get("MAX_SEARCH_PAGES_INFER", 4)
    MAX_LINKS_VECTORDB = config.get("MAX_LINKS_VECTORDB", 3)
    MAX_LINKS_SEARCHAPI = config.get("MAX_LINKS_SEARCHAPI", 3)

##############################################################
# Truncate Query for searching 
##############################################################
def truncate_query(text, max_length=MAX_SEARCH_QUERY_LENGTH):
    return text[:max_length] if len(text) > max_length else text


##############################################################
# Extract words from user query for searching 
##############################################################
def extract_keywords_for_search(user_input):
 
    #additional_fluff = []
    additional_fluff = {"team","teams","last","past","please","summary", "total","sentence","word","sentences","words"}
    user_input = user_input.lower()

    # Load English stopwords
    stop_words = set(stopwords.words('english')).union(additional_fluff)

    # Tokenize the input
    words = word_tokenize(user_input)
    
    # Filter out stopwords and non-alphanumeric words
    keywords = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    filterd_words = " ".join(keywords)
    print(f"Filtered words: {filterd_words}")

    # keywords
    doc = nlp(filterd_words)
    #
    # Filter out irrelevant parts of speech like AUX (auxiliary verbs), DET (determiners), etc.
    keywords = [
        token.text for token in doc
        if token.pos_ not in {"AUX", "DET", "PRON", "CCONJ", "PART","VERB"} and token.ent_type_ not in {"DATE", "TIME"} and token.is_alpha
    ]
    filterd_words = " ".join(keywords)

    return filterd_words

##############################################################
# Token Counter (only rough count as we're skipping AutoTokenizer/limits on model)
##############################################################
def count_tokens(text):
    tokenizer = AutoTokenizer.from_pretrained(config_embedding_model_name)
    tokens = tokenizer(text)
    token_count = len(tokens["input_ids"])
    return token_count

##############################################################
# Token Counter (only rough count as we're skipping AutoTokenizer/limits on model)
##############################################################
def count_tokens_approx(text):
    text = text.lower().strip()  # Normalize case
    words = text.split()  # Split by whitespace
    subword_count = 0

    for word in words:
        # Approximate how WordPiece splits words into subwords
        if len(word) > 6:  # Longer words likely split into multiple subwords
            subword_count += len(re.findall(r"[aeiouy]+|[^aeiouy]+", word))  # Rough subword split
        else:
            subword_count += 1  # Short words usually map to one token

    return subword_count

##############################################################
# Search pages per Query
##############################################################
def search_confluence_pages(query: str):
    headers = {
        "Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"
    }
    params = {
        'cql': f'siteSearch ~ "{query}" AND space in ("ADH") AND type in ("page")',
        'start': 0,
        'limit': 20,
        'includeArchivedSpaces': 'false'
    }

    response = requests.get(f"{CONFLUENCE_REST_URL}/content/search",headers=headers,params=params)
    pagesList = response.json().get("results")
        
    return pagesList


##############################################################
# Get Page Details per PageID
##############################################################
def get_page_details(page_id):
    headers = {
        "Authorization": f"Bearer {CONFLUENCE_API_TOKEN}"
    }

    #print(f"Incoming pageId to get content: {page_id}")
    response = requests.get(f"{CONFLUENCE_REST_URL}/content/{page_id}?expand=body.storage",
                             headers=headers)
    return response.json()

##################################################################
# Convert HTML to Text
##################################################################
def html_to_cleantext(html_content: str):

 # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract text using a space as a separator to retain spacing between block elements
    text = soup.get_text(separator=" ", strip=True)
    
    # Normalize the text to NFKD form and encode it to ASCII, ignoring non-ASCII characters.
    normalized_text = unicodedata.normalize("NFKD", text)
    ascii_text = normalized_text.encode("ascii", "ignore").decode("ascii")
    return ascii_text

###################################################
# Retrieve relevant context from Vector
###################################################
def retrieve_context(query: str, search_type: str) -> Optional[str]:
        
        try:

            context_metadata: list[dict] = [] 
            context_texts: list[str] = []
            aiVectorPagesList=[]
            lowRankPagesList=[]
            numHighPagesRelevant=0
            
            #########################################################
            ### AI Search Section
            #########################################################
            if (search_type.startswith("ai")):
                print("Performing AI stuff....")
                
                query_vector = embedding_model.embed_query(query)

                #print("******  Searching vector store...")
                retrieved_docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                    embedding=[query_vector],
                    k=MAX_NUM_CHUNKS_RETURN,
                )

                print(f"Number of total relevance pages found by AI search: {len(retrieved_docs)}")
                
                vectorCollection = vectorstore._collection

                ### Loop through all relevant pages from AI search engine
                for doc, score in retrieved_docs:

                    pageInfo = f"Title: {doc.metadata['title']}, PageId: {doc.metadata['pageid']}, Score: {score}"
                   
                    ### Get page content for checking content size
                    pageLookup = vectorCollection.get(where={"pageid": doc.metadata['pageid']})
                    combinedPageContent = " ".join(pageLookup.get("documents", []))
                    

                    #print("===========================================================================================")
                    #print(f"*** Length of vector Content: {len(pageContent)} for Page Info: {pageInfo}")
                    #print(f"*** Length of entire page1: {len(docsstuff)} for Page Info: {pageInfo}")
                    #print(f"*** Length of entire page: {len(combinedPageContent)} for Page Info: {pageInfo}")
                    #print(f"-------------------------------")
                    #print(f"Actual page content: {combinedPageContent}")
                    #print("===========================================================================================")
                    
                    ### Pick up high-ranked pages first
                    if (score > MAX_SCORE_THRESHOLD or score < MIN_SCORE_THRESHOLD):
                        print(f"Skipping page due to low score , {score} - Page: {pageInfo} ")
                        if (doc.metadata['pageid'] not in lowRankPagesList):
                            lowRankPagesList.append(doc.metadata['pageid'])
                    else:      
                        if doc.metadata['pageid'] not in aiVectorPagesList and len(combinedPageContent) > MIN_PAGE_CONTENT_SIZE_CHARS and len(aiVectorPagesList) < MAX_INFER_PAGES and numHighPagesRelevant < MAX_HIGH_RELEVANT_PAGES:
                            numHighPagesRelevant = numHighPagesRelevant + 1
                            aiVectorPagesList.append(doc.metadata['pageid'])
                            print(f"Including HIGH ranked page - {pageInfo}, Content size: {len(combinedPageContent)} - Current Num of relevant pages: {numHighPagesRelevant}")
                        else:
                            print(f"Skipping page due page already exists in list or reached max number of relevant pages to use - {pageInfo} ")
                            
                print(f"Number of included HIGH relevant pages: {numHighPagesRelevant}")

                # If List does not have enough "high" relevant pages, include some low ones too
                numLowRankPages=0            
                if (len(aiVectorPagesList) < MAX_INFER_PAGES and len(aiVectorPagesList) < MAX_AI_RELEVANT_PAGES):
                    # Loop through the low ranks and include the top ones only
                    for pageId in lowRankPagesList:
                    
                            ### Get page content for checking content size
                            pageLookup = vectorCollection.get(where={"pageid": pageId})
                            combinedPageContent = " ".join(pageLookup.get("documents", []))
                    
                            if (len(aiVectorPagesList) < MAX_INFER_PAGES and len(combinedPageContent) > MIN_PAGE_CONTENT_SIZE_CHARS and numLowRankPages < MAX_LOW_RELEVENT_PAGES and pageId not in aiVectorPagesList):
                                numLowRankPages = numLowRankPages + 1
                                print(f"Including LOW ranked page: {pageId}, Content size: {len(combinedPageContent)}")
                                aiVectorPagesList.append(pageId)
                print(f"Number of included LOW relevant pages: {numLowRankPages}")

                ### Build list of context containing both clean text/data and metadata
                for index,pageId in enumerate(aiVectorPagesList):
                    
                    #print(f"CONTEXT BUILD - Vector PageId: {pageId}")

                    retrieved_docs = vectorCollection.get(where={"pageid": pageId})
                    docs = retrieved_docs.get("documents", [])  # list of chunk texts
                    combined_document = " ".join(docs)
                    #print(f"combined_document: {combined_document}")

                    # Add metadata for Source Links in UI
                    if (index < MAX_LINKS_VECTORDB):
                        metadatas = retrieved_docs.get("metadatas", [])
                        metadata = metadatas[0] if metadatas else {}
                        context_metadata.append(metadata)

                    #print(f"CONTEXT BUILD - Vector Metadata: {metadata}")
                    
                    # Add document for AI inference
                    context_texts.append(combined_document)
                    

            print(f"Total number of AI pages ready for infer: {len(aiVectorPagesList)}")

            #################################################################################################
            # Retrieve pages from Confluence Search API - Fill in rest with Search pages if available slots
            #################################################################################################
            numIncludedSearchPages=0
            numPagesRelevant = len(aiVectorPagesList)
            searchQuery = truncate_query(query)
            searchKeywords = extract_keywords_for_search(searchQuery)
            print(f"**SEARCH*** - Original query: {query}")
            print(f"**SEARCH*** - Filtered Search Keywords (truncated): {searchKeywords}")
            searchPagesList = search_confluence_pages(searchKeywords)
            for index, searchPage in enumerate(searchPagesList):
                pageid = searchPage["id"]
                if (   ( not search_type.startswith("ai") and numPagesRelevant < 10 )   or  (search_type.startswith("ai") and numPagesRelevant < MAX_INFER_PAGES and numIncludedSearchPages < MAX_SEARCH_PAGES_INFER and pageid not in aiVectorPagesList)):
                    numPagesRelevant = numPagesRelevant + 1
                    numIncludedSearchPages = numIncludedSearchPages + 1
                    
                    title = searchPage["title"]
                    print(f"*** SEARCH PAGE: **** PageId and Title: {pageid}, {title}")
                    pageDetails = get_page_details(pageid)
                    body_value = pageDetails["body"]["storage"]["value"]
                    cleanSearchPage = html_to_cleantext(body_value)

                    # Add metadata for Source Links in UI
                    if ( not search_type.startswith("ai") or (index < MAX_LINKS_SEARCHAPI)):
                        searchMetadata = {
                            "pageid": pageid,
                            "source": SEARCH_SOURCE_PAGE,
                            "title": title
                        }
                        context_metadata.append(searchMetadata)
                    
                    # Add document for AI inference
                    context_texts.append(cleanSearchPage)
                    
                    #print(f"======================================================================================================")
                    #print(f"PageBody: {body_value}")
                    #print(f"======================================================================================================")
            print(f"Number of included SEARCH pages: {numIncludedSearchPages}")

            print(f"Total number of ALL pages (both AI and Search) ready for infer: {len(context_texts)}")

            return " \n ".join(context_texts), context_metadata
        
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return None


##############################################################
# Get conversation history per User
##############################################################
def get_user_conversation_history(user_id: str):

    ### DISABLE CHAT HISTORY FOR THIS APPLICATION - We do not want to build history as this will overload the LLM / not needed at this time
    ### REMOVE THIS LINE BELOW TO ENABLE CHAT HISTORY
    if not ENABLE_CHAT_CONVERSATION:
        print("Chat conversation is disabled")
        chat_histories = {}
    else:
        print("Enable chat conversation - will track all conversations per user")
    
    if user_id not in chat_histories:
        # Append system / overall instructions to history (use 'system' as role)
        chat_histories[user_id] = [{"role": "system", "content": "You are a helpful AI assistant in summarizing content."}]
    return chat_histories[user_id]

##############################################################
# Streams the AI's response 
##############################################################
async def stream_ollama_response(user_id: str, user_input: str, vectorContext: list[str], metadataContext: list[dict], search_type: str, infer_start_time):
        
    # Get or create conversation history for the user
    chat_history = get_user_conversation_history(user_id)
    
    # Build full query/input for LLM processing
    if vectorContext and search_type.startswith("ai"):

        fullPrompt = f"""### Task: 
            Respond to the user query using the provided context

            ### Guidelines:
            - If you don't know the answer, clearly state that.
            - If uncertain, ask the user for clarification.
            - Respond in the same language as the user's query.
            - If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
            - If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
            - Do not use XML tags in your response.
            - Ensure citations are concise and directly related to the information provided.

            ### Output:
            Provide a clear and direct response to the user's query,

            <context>
            {vectorContext}
            </context>

            <user_query>
            {user_input}
            </user_query>
            """
        

        chat_history.append({"role": "user", "content": fullPrompt})
        
        # Call Ollama API for inference with streaming enabled
        API_CHAT=AI_OLLAMA_SERVER+ ":" + AI_OLLAMA_PORT + "/api/chat"
        response = requests.post(API_CHAT, json={
            "model": config_ollama_model_name,
            "messages": chat_history,
            "stream": True
        }, stream=True)

    else:
        chat_history.append({"role": "user", "content": "Summarize my search keywords: " +  user_input + ", in a quick few sentences at most."})
    
    #print(f"=====================================================================================")
    #print(f"Chat History: {chat_history}")
    #print(f"=====================================================================================")
    
    # Stream the response back to the client
    def event_generator():
        infer_end_time = time.time()
        infer_duration = infer_end_time - infer_start_time

         # First Chunk - Metadata - List of links
        metadataList = {
                "metadataList": metadataContext
        }
        # Sending metadata as the first chunk
        yield json.dumps(metadataList) + "\n"

        aiResponse = ""

        if (search_type.startswith("ai")):
            # Data Stream
            for line in response.iter_lines():
           
                body = json.loads(line)
                response_part = body["message"]["content"]
                aiResponse += response_part
                #print(response_part, end='', flush=True)
                #print(body, end='', flush=True)
                yield response_part

                response_done = body["done"]
                if (response_done is True):
                    print("End of data stream", flush=True)
                    ui_complete_end_time = time.time()
                    ui_complete_duration = ui_complete_end_time - infer_start_time
                    print(f"Inference time: {infer_duration:.4f} seconds , Inference + UI completion time: {ui_complete_duration:.4f}", flush=True)


            # Append response to history (use 'assistant' as role)
            chat_history.append({"role": "assistant", "content": aiResponse})

        print("=======================================================================================")
        ### Token Counter
        combinedtext = " ".join(item["content"] for item in chat_history)
        num_tokens_approx = count_tokens_approx(combinedtext)
        #num_tokens = count_tokens(combinedtext)
        print(f"Initiate inference process - Length of input: {len(combinedtext)} characters, Number of words: {len(combinedtext.split())} words, Approximate # of input tokens <= {num_tokens_approx*TOKEN_NUM_REDUCER}   ")
        print("=======================================================================================")
       
    return StreamingResponse(event_generator(), media_type="text/event-stream")


##############################################################
# CHAT Endpoint
##############################################################
@app.post("/api/chat-stream")
async def chat(request: Request):
    
    # Dynamically load in AI config per request
    load_ai_config()

    print(f"Inference settings - MAX_INFER_PAGES: {MAX_INFER_PAGES}, MAX_AI_RELEVANT_PAGES: {MAX_AI_RELEVANT_PAGES}, MAX_HIGH_RELEVANT_PAGES: {MAX_HIGH_RELEVANT_PAGES}, MAX_SEARCH_PAGES_INFER: {MAX_SEARCH_PAGES_INFER}")

    data = await request.json()
    user_id = data.get("user_id", "default_user")  # Get user_id from user session
    search_type = data.get("searchType")
    print(f"Search type: {search_type}")
    user_input = data.get("query")
    if not user_input:
        print("Empty query")
        return {"error": "Empty query"}

    print("---------------------------------------------------------------------------------------")
    start_time = time.time()
    print(f"Starting 'retrieve_context' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
    # Retrieve context and generate response
    vectorContext, metadataContext = retrieve_context(user_input, search_type)
    if vectorContext is None:
        print("Error retrieving context. Please try again.")
        return {"error": "Error retrieving context"}

    end_time = time.time()
    retrieve_context_duration = end_time - start_time
    print(f"Finished 'retrieve_context' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Execution time for 'retrieve_context': {retrieve_context_duration:.4f} seconds")
    print("---------------------------------------------------------------------------------------")

    if (not search_type.startswith("ai")):
        print(f"Search is: {search_type} - Empty out vectorContext")         
        vectorContext = [] 

    infer_start_time = time.time()
    return await stream_ollama_response(user_id, user_input, vectorContext, metadataContext, search_type, infer_start_time)

            
# You can run the server with:
# uvicorn aiwebservice:app --host 0.0.0.0 --port 8000
# uvicorn aiwebservice:app --host 0.0.0.0 --port 82
#
# uvicorn aiwebservice:app --reload
        
