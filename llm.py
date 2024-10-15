from haystack.nodes import PreProcessor, PromptModel, PromptTemplate, PromptNode
#preprocessor is used to handle the data before feeding it to the model
#PromptTemplate is used to define the prompt that will be used to query the model
#PromptNode is used to define the model that will be used to query the data

from haystack import Document
#Document is used to define the data that will be used to query the model

from haystack.document_stores import InMemoryDocumentStore
#InMemoryDocumentStore is used to store the data in memory

from haystack import Pipeline
#Pipeline is used to define the pipeline that will be used to query the model

from haystack.nodes import BM25Retriever
#BM25Retriever is used to retrieve the data from the document store

from pprint import pprint
from json import loads, dumps
import numpy as np
import pdfplumber

#API key for huggingface
HF_TOKEN = "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" 

#loading the data from the json file i.e. Rag Database
# with pdfplumber.open('Xenia Rulebook.pdf') as pdf:
#     # data = loads(f.read())
#     text = ''
#     for page in pdf.pages:
#         text += page.extract_text()/
with open("info.txt",'r') as f :
    text = f.read()


#splitting the data into 'n' parts
# docs = np.array_split(data, 5)
docs = np.array_split([text], 50)

print(docs)
docs = [str(doc.tolist()) for doc in docs]
docs = [Document(content=doc) for doc in docs]

#preprocessing the data
processor = PreProcessor()
ppdocs = processor.process(docs)

#storing the data in memory
docu_store = InMemoryDocumentStore(use_bm25=True)
docu_store.write_documents(ppdocs)
retriever = BM25Retriever(docu_store, top_k = 4) #retriever to retrieve the data from the document store k = no. of documents to retrieve

#prompt model to query the data
qa_template = PromptTemplate(
    prompt =
    '''
    You are an AI chatbot your task is to give info about the events.
    Provide info from data provided to you. Don't provide extra information.                                      
    Context: {join(documents)};
    Prompt: {query}
    '''
)

prompt_node = PromptNode(
    model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1",  #model to query
    api_key = HF_TOKEN,                                           #api key for huggingface
    default_prompt_template=qa_template,                          #prompt template to query the model
    max_length = 2500,                                            #max length of the output
    model_kwargs={"model_max_length":20000}                       #max length of the input
)

#pipeline to query the model
rag_pipeline = Pipeline()
rag_pipeline.add_node(component=retriever, name = 'retriever', inputs=['Query'])
rag_pipeline.add_node(component=prompt_node, name = 'prompt_node', inputs=['retriever'])


#function to return the answer
def return_ans(q):
    try:
        ans = rag_pipeline.run(query = q)
        response = {
            "data" : ans['results'][0],
            "status":200
        }
        return response
    except Exception as e:
        response = {
            "data" : e,
            "status" : 500
        }
        return response

#to test the model

# q = f"team size of concepts event?"
# ans = rag_pipeline.run(query = q)

# print(type(ans['results']))

# for i in ans['results']:
#     print(i.strip())

# while 1:
    # q = input("Enter query: ")
    # ans = rag_pipeline.run(query = q)
    # print(ans['results'][0])