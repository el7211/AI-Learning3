from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from supabase import create_client
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set up Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_api_key = os.getenv("AZURE_API_KEY")
supabase = create_client(url, key)

# This deletes all rows in documents where id is not equal None
supabase.table("documents").delete().not_.is_("id", None).execute()



# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Initialize vector store with your Supabase client and embedding function
vector_db = SupabaseVectorStore(
    client=supabase,
    embedding=embedding_model,
    table_name="documents"
)


# try to chunk the PDF
loader = PyPDFLoader("docs/ITGettingStartedGuide.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
vector_db.add_documents(chunks)


'''
query="What is Intouch symbol?"
# Run a similarity search
results = vector_db.similarity_search(query=query, k=1)
# Print the top k chunks
for i, doc in enumerate(results, 1):
    print(f"\n--- Result #{i} ---")
    print(doc.page_content)
'''

print("==========Response after LLM: ===================")
# Get user query and retrieve relevant chunks in vector_db
user_input = input("Ask the assistant anything about Intouch: ").strip()
retriever = vector_db.as_retriever()
relevant_docs = retriever.invoke(user_input)
context = "\n\n".join(doc.page_content for doc in relevant_docs)

# Set Azure model and prompts, in the prompt, insert data chunks as context
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



