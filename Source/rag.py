import chromadb
import os
import asyncio
from pathlib import Path
from enum import Enum
from docx2python import docx2python
from PyPDF2 import PdfReader
import nest_asyncio
from azure.storage.blob import BlobServiceClient
import sys
import rag_llm_wrapper
import rag_text_helper

nest_asyncio.apply()

# Avoid re-entrance complaints from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

COLLECTION_NAME = 'chat-doc-folder'
USER_PROMPT = "Please enter your question (or type 'done' to finish): "
MAX_BATCH_SIZE = 5461  # Set to the maximum allowed batch size

class vector_store:
    '''Encapsulates Chroma the vector store and its parameters (e.g. for doc chunking)'''
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_client = chromadb.Client()
        self.coll = self.chroma_client.get_or_create_collection(name='chat_doc_folder')
        self.id_counter = 0

    def text_split(self, text):
        return rag_text_helper.text_split_fuzzy(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator='\n')

    def new_docid(self):
        self.id_counter += 1
        nid = f'id-{self.id_counter}'
        return nid

    def update(self, chunks, metas):
        for i in range(0, len(chunks), MAX_BATCH_SIZE):
            batch_chunks = chunks[i:i + MAX_BATCH_SIZE]
            batch_metas = metas[i:i + MAX_BATCH_SIZE]
            ids = [self.new_docid() for _ in batch_chunks]
            self.coll.add(documents=batch_chunks, ids=ids, metadatas=batch_metas)

    def search(self, q, limit=None):
        results = self.coll.query(query_texts=[q], n_results=limit)
        print(results['documents'][0][0][:100])
        return results['documents'][0]

def read_word_doc(fpath, store):
    '''Convert a single word doc to text, split into chunks & add these to vector store'''
    try:
        with docx2python(fpath) as docx_content:
            doctext = docx_content.text
        chunks = list(store.text_split(doctext))
        metas = [{'source': str(fpath)}]*len(chunks)
        store.update(chunks, metas=metas)
    except Exception as e:
        print(f"Error processing Word document {fpath}: {e}")

def read_pdf_doc(fpath, store):
    '''Convert a single PDF to text, split into chunks & add these to vector store'''
    try:
        pdf_reader = PdfReader(fpath)
        doctext = ''.join((page.extract_text() for page in pdf_reader.pages))
        chunks = list(store.text_split(doctext))
        metas = [{'source': str(fpath)}]*len(chunks)
        store.update(chunks, metas=metas)
    except Exception as e:
        print(f"Error processing PDF {fpath}: {e}")

async def async_main(oapi, docs, verbose, limit, chunk_size, chunk_overlap, question):
    store = vector_store(chunk_size, chunk_overlap)
    
    for fname in docs.iterdir():
        if fname.is_file():
            #print(f"Processing file: {fname} ({fname.suffix})")
            if fname.suffix in ['.doc', '.docx']:
                read_word_doc(fname, store)
            elif fname.suffix == '.pdf':
                read_pdf_doc(fname, store)
            else:
                print(f"Skipping unsupported file type: {fname}")

    done = False
    while not done:
        print('\n')
        if question:
            user_question = question
            question = None  # Reset to allow for interactive input
        else:
            user_question = input(USER_PROMPT)
        if user_question.strip().lower() == 'done':
            break

        docs = store.search(user_question, limit=limit)
        if verbose:
            print(docs)
        if docs:
            gathered_chunks = '\n\n'.join(docs)
            sys_prompt = '''\
You are a helpful assistant, who answers questions directly and as briefly as possible.
Consider the following context and answer the user's question.
If you cannot answer with the given context, just say so.\n\n'''
            sys_prompt += gathered_chunks + '\n\n'
            messages = rag_llm_wrapper.prompt_to_chat(user_question, system=sys_prompt)
            if verbose:
                print('-'*80, '\n', messages, '\n', '-'*80)

            model_params = dict(
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=1,
                temperature=0.1
            )

            retval = await oapi(messages, **model_params)
            if verbose:
                print(type(retval))
                print('\nFull response data from LLM:\n', retval)

            print('\nResponse text from LLM:\n\n', retval.first_choice_text)
    return

def blob(path):
    connection_string = 'DefaultEndpointsProtocol=https;AccountName=your-account-name;AccountKey=your-account-key'
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = 'avengercontainer'
    container_client = blob_service_client.get_container_client(container_name)

    os.makedirs(path, exist_ok=True)

    for blob in container_client.list_blobs():
        blob_name = blob.name
        blob_client = container_client.get_blob_client(blob_name)
        download_file_path = os.path.join(path, blob_name)
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

        with open(download_file_path, 'wb') as download_file:
            download_file.write(blob_client.download_blob().readall())

        #print(f"Downloaded {blob_name} to {download_file_path}")
    return Path(path)

def main(verbose, chunk_size, chunk_overlap, limit, openai_key, apibase, model, docs, question=None):
    if openai_key:
        oapi = rag_llm_wrapper.openai_chat_api(api_key=openai_key, model=(model or 'gpt-35-turbo-16k'))
    else:
        oapi = rag_llm_wrapper.openai_chat_api(model=model, base_url=apibase)

    asyncio.run(async_main(oapi, docs, verbose, limit, chunk_size, chunk_overlap, question))

# Manually setting the arguments (as you would pass them from the command line)
verbose = False
chunk_size = 200
chunk_overlap = 20
limit = 4
os.environ['AZURE_OPENAI_ENDPOINT'] = 'Your-OpenAI-API key'
os.environ['AZURE_OPENAI_API_KEY'] = 'Your-OpenAI-API Base'
os.environ['OPENAI_API_VERSION'] = '2024-06-01'
openai_key = 'Your-OpenAI-API key'
apibase = 'Your-OpenAI-API Base'
model = 'gpt-4o'
path = '/home/avenger/docs'
question = None  # Set this to a specific question if needed, otherwise it will prompt

docs_folder = blob(path)
sys.path.append('/home/avengers')

# Call the main function with the arguments
main(verbose, chunk_size, chunk_overlap, limit, openai_key, apibase, model, docs_folder, question)
