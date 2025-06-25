from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_api_key = os.getenv("AZURE_API_KEY")

# Step 1: Load the PDF
loader = PyPDFLoader("docs/ElaineLiao.pdf")
docs = loader.load()

# Step 2: Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 3: Embed with Hugging Face model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Store in ChromaDB (local vector store)
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Step 5: Get user query and retrieve relevant chunks in vector_db
user_input = input("Ask the assistant something about Elaine’s CV: ").strip()
retriever = vector_db.as_retriever()
relevant_docs = retriever.invoke(user_input)
context = "\n\n".join(doc.page_content for doc in relevant_docs)

# Step 6: Set Azure model and prompts, in the prompt, insert data chunks as context
model = AzureChatOpenAI(
    openai_api_version="2024-12-01-preview",
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name="gpt-4.1-mini-Yeo",  # This must match your Azure deployment name exactly
)

system_prompt = "You are an assistant that uses the provided context to answer questions accurately."
full_prompt = f"""{system_prompt}

Context:
{context}

Question: {user_input}

Answer:"""


# Step 7: Final output
response = model.invoke(full_prompt)
print("\n🧠 Assistant:\n", response.content)










