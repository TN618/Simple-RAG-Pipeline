
'''
# important installations
pip install PyPDF2
pip install langchain-text-splitters
pip install langchain-community sentence-transformers
pip install faiss-cpu
pip install langchain-google-genai google-generativeai

'''


import json

# 1. Specify the file path to your .ipynb notebook file.
notebook_path = '/notebook.ipynb'

# Initialize a list to store the extracted content
notebook_content = []

try:
    # 2. Open and load the .ipynb file as a JSON object.
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # 3. Iterate through each cell in the loaded notebook.
    for cell in notebook['cells']:
        # 4. For each cell, check its type. If it is a 'code' cell or a 'markdown' cell, extract its 'source' content.
        if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown':
            # 5. The 'source' content for each cell is typically a list of strings (lines of code or markdown).
            # Join these strings to form a single text block for that cell.
            source_content = ''.join(cell['source'])
            notebook_content.append(source_content)

    # 6. Concatenate all extracted text blocks from the code and markdown cells into one comprehensive string,
    # or store them as a list of strings, to represent the entire notebook's content.
    # We are storing them as a list of strings as per the subtask description.
    print(f"Successfully extracted content from '{len(notebook_content)}' code/markdown cells.")
    print("First 500 characters of the combined notebook content (for verification):")
    print('\n'.join(notebook_content)[:500])

except FileNotFoundError:
    print(f"Error: The file '{notebook_path}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{notebook_path}'. Make sure it's a valid .ipynb file.")
except KeyError as e:
    print(f"Error: Missing expected key in the notebook structure: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


from PyPDF2 import PdfReader

# 1. Specify the file path to your .pdf file.
pdf_path = '/document.pdf'  # Updated to the correct existing PDF file path

# Initialize an empty string to store the extracted text
pdf_content = ''

try:
    # 2. Open the PDF file in binary read mode ('rb').
    with open(pdf_path, 'rb') as f:
        # 3. Create a PdfReader object from the opened file.
        reader = PdfReader(f)

        # 4. Iterate through each page of the PDF document.
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            # 5. Extract the text content from each page.
            text = page.extract_text()
            if text:
                # Append the extracted text, adding a newline for separation between pages
                pdf_content += text + '\n'

    # 6. Print a confirmation message and display the first few hundred characters.
    print(f"Successfully extracted content from '{len(reader.pages)}' pages of the PDF.")
    print("First 500 characters of the extracted PDF content (for verification):")
    print(pdf_content[:500])

except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found.")
except Exception as e:
    print(f"An error occurred while processing the PDF: {e}")


combined_notebook_content = "\n".join(notebook_content)
combined_text = combined_notebook_content + "\n\n" + pdf_content

print("Combined text length:", len(combined_text))
print("First 500 characters of combined text:\n", combined_text[:500])

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Re-define combined_text if it's also not in scope (assuming it was defined earlier)
# For this fix, we assume combined_text is already defined from previous steps
# If combined_text is also not defined, the previous steps need to be re-run first.

# 1. Re-initialize RecursiveCharacterTextSplitter with suitable chunk_size and chunk_overlap.
# This ensures 'chunks' is defined in this execution context.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Example chunk size
    chunk_overlap=200 # Example chunk overlap
)

# 2. Re-use the split_text() method to divide the combined text into chunks.
chunks = text_splitter.split_text(combined_text)

# 3. Initialize an embedding model using HuggingFaceEmbeddings.
# 'all-MiniLM-L6-v2' is a commonly used and efficient model.
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# 4. Initialize the FAISS vector database from the generated chunks and the embedding model.
vector_db = FAISS.from_texts(chunks, embeddings)

# Print a confirmation message
print("FAISS vector database initialized successfully.")
print(f"Number of vectors in the database: {vector_db.index.ntotal}")

import os

# Define a directory to save the FAISS index
save_directory = "./faiss_index"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the vector_db to the specified directory
vector_db.save_local(save_directory)

print(f"FAISS vector database saved to '{save_directory}' successfully.")

import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import textwrap

# Configure the generativeai library with your API key
# This assumes GOOGLE_API_KEY is already set in your Colab secrets or environment variables
# If not, please set it in Colab's 'Secrets' tab on the left sidebar under the key GOOGLE_API_KEY.
# Or uncomment the line above and replace 'YOUR_API_KEY' with your actual key.
if 'GOOGLE_API_KEY' not in os.environ:
    print("Error: GOOGLE_API_KEY not found in environment variables. Please set it in Colab secrets or directly in the code.")
else:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("Google Generative AI configured successfully.")

# Re-initialize the Medium language model with 'gemini-1.0-pro'.
# 'gemini-1.0-pro' is generally more robust for text generation via the API.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

print("Google gemini-2.5-pro language model re-initialized successfully.")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Define the directory where the FAISS index was saved
save_directory = "./faiss_index"

# 2. Initialize an embedding model using HuggingFaceEmbeddings with the same model_name.
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# 3. Load the FAISS vector database from the specified directory.
# The allow_dangerous_deserialization=True is added to enable deserialization.
loaded_vector_db = FAISS.load_local(save_directory, embeddings, allow_dangerous_deserialization=True)

# 4. Print a confirmation message and optionally display the number of vectors.
print(f"FAISS vector database loaded from '{save_directory}' successfully.")
print(f"Number of vectors in the loaded database: {loaded_vector_db.index.ntotal}")

# Re-create the RAG chain with the updated LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Convert the loaded_vector_db into a retriever.
# Assuming 'loaded_vector_db' is still in scope from previous successful execution.
# If not, it needs to be re-loaded or re-created.
# For this fix, we assume loaded_vector_db is still valid.
retriever = loaded_vector_db.as_retriever()

# Create a prompt template for the RAG chain.
prompt_template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RAG chain.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
print("RAG chain re-created successfully with gemini-2.5-pro.")

# 1. Define a sample question to query the RAG chain.
question = "How did the project tackles the problem of missing data?"

# 2. Invoke the qa_chain with the sample question to get an answer.
# The qa_chain expects a dictionary with 'query' key for the question.
response = qa_chain.invoke({"query": question})

# 3. Print the question and the generated answer to verify the RAG chain's functionality.
print(f"Question: {question}")
print("\nAnswer:")
# Use textwrap to format the answer for better readability
print(textwrap.fill(response['result'], width=80))
