import os
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import  PyPDFLoader 
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")
from run_local import initialize_llm
import os

embedding_model_path= "BAAI_bge-base-en-v1.5"

if not os.path.exists(embedding_model_path):
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder=".")
else:
    pass


embeddings = HuggingFaceEmbeddings(model_name= embedding_model_path,
                                    model_kwargs = {'device':'cpu'},
                                   encode_kwargs = {'normalize_embeddings': True})



chat_history = []

#Load the PDF File

def load_file(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document

# Splitting the file and store it into vector DB

def chunking_vectordb(document):
    ## Split documnets into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000,
                                                   chunk_overlap= 200)
    text_chunk = text_splitter.split_documents(document)
    #Load the Embedding Model
    #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    vector_store = FAISS.from_documents(text_chunk, embeddings)
    return vector_store

template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow up Input: {question}
Standalone questions: """

CONDENSE_QUESTION_PROMPT = PromptTemplate(template= template, input_variables= ["question"])


application = Flask(__name__)
app = application

## Route for a home page
@app.route("/")
def index():
    return render_template("first.html")


@app.route("/start", methods= ["GET", "POST"])

def start():
    if request.method == "POST":
        os.makedirs("data", exist_ok=True)
        file= request.files["file"]
        print(file)
        if file:
            file_path = os.path.join("data/" + secure_filename(file.filename))
            file.save(file_path)
            document = load_file(file_path)
            vector_store = chunking_vectordb(document)
            vector_store.save_local("faiss")
            return render_template("index.html")

@app.route("/get_answer", methods= ["GET", "POST"])

def get_answer():
    if request.method == "POST":
        user_input = request.form["question"]
        llm = initialize_llm()
        store = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization= True)
        chain = ConversationalRetrievalChain.from_llm(llm= llm,retriever=store.as_retriever(search_kwargs={'k': 2}),
                                               condense_question_prompt=CONDENSE_QUESTION_PROMPT,return_source_documents=True, 
                                               verbose=False)
        
        result = chain.invoke({"question": user_input, "chat_history": chat_history})
        
        chat_history.extend(
            [
                HumanMessage(content= user_input),
                AIMessage(content=result["answer"])
            ]
        )
        print(f"Answer: {result['answer']}")
        print(chat_history)

    return render_template("index.html", results = str(result['answer']))
        

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)