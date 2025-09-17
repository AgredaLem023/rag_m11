import os
import numpy as np
import chromadb
from sentence_transformers import CrossEncoder
from helper_utils import word_wrap, load_chroma
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chonkie import SentenceChunker

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

embedding_function = SentenceTransformerEmbeddingFunction()


reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]



# Use Chonkie's SentenceChunker for sentence-aware, token-bounded chunking
doc_text = "\n\n".join(pdf_texts)

chunker = SentenceChunker(
    tokenizer_or_token_counter="gpt2",
    chunk_size=512,
    chunk_overlap=50,
    min_sentences_per_chunk=1,
)

chunks = chunker.chunk(doc_text)
token_split_texts = [c.text for c in chunks]

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    "microsoft-collect", embedding_function=embedding_function # type: ignore
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def generate_multi_query(query, model="gpt-3.5-turbo"):

    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n") # type: ignore
    return content

original_query = (
    "What details can you provide about the factors that led to revenue growth?"
)
generated_queries = generate_multi_query(original_query)

# concatenate the original query with the generated queries
queries = [original_query] + generated_queries


results = chroma_collection.query(
    query_texts=queries, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents: # type: ignore
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

scores = cross_encoder.predict(pairs)

# ====================================================

# Get the top 5 documents based on the scores
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]

# Concatenate the top documents into a single context
context = "\n\n".join(top_documents)


# Generate the final answer using the OpenAI model
def generate_answer_from_multi_query(query, context, model="gpt-3.5-turbo"):

    prompt = f"""
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"based on the following context:\n\n{context}\n\nAnswer the query: '{query}'",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages, # type: ignore
    )
    content = response.choices[0].message.content
    content = content.split("\n") # type: ignore
    return content


res = generate_answer_from_multi_query(query=original_query, context=context)
print("Answer:")
print(res)
