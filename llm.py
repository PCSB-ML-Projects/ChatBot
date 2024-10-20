from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

embeddings = HuggingFaceEmbeddings(model_name= "BAAI/bge-large-en-v1.5")

docs = ""
with open("info.txt", 'r') as f:
    docs = f.read()
    docs = docs.split('???')

for i in range(len(docs)):
    docs[i] = Document(page_content=docs[i])

embeddings = embeddings
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

client = InferenceClient(api_key="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")

def ask(query: str) -> str:
    ret = retriever.invoke(query)
    system_prompt = f'''You are an AI assistant named XenoBot that helps people find information.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
    You are a chatbot of PICT CSI CLUB which answer queries and recommend competitions of the club's flagship event "XENIA".
    MAKE SURE YOU FULLY ANSWER THE QUERY and use indentation.
    Give a generic answer if the query is not found in the data.
    Data : <{ret[0].page_content}>,
           <{ret[1].page_content}>,
           <{ret[2].page_content}>     
    '''
    messages = [
        ("system", system_prompt),
        ("user", f"{query}"),
    ]
    message = client.chat_completion(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=messages,
        max_tokens=1000,
        temperature=0.2,
        stream=False
    )
    return message.choices[0].message.content

while 1:
    query = input("Ask query : ")
    response = ask(query)
    print(response)