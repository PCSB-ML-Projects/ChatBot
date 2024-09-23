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

#API key for huggingface
HF_TOKEN = "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" 

#loading the data from the json file i.e. Rag Database
with open('RagData.json', 'r') as f:
    data = loads(f.read())


#splitting the data into 'n' parts
docs = np.array_split(data, 50)
docs = [str(doc.tolist()) for doc in docs]
docs = [Document(content=doc) for doc in docs]

#preprocessing the data
processor = PreProcessor()
ppdocs = processor.process(docs)

#storing the data in memory
docu_store = InMemoryDocumentStore(use_bm25=True)
docu_store.write_documents(ppdocs)
retriever = BM25Retriever(docu_store, top_k = 1) #retriever to retrieve the data from the document store k = no. of documents to retrieve


#prompt model to query the data
qa_template = PromptTemplate(
    prompt =
    '''
    <PrePrompt>                                                   # This is a prompt to query the model to what to do before the user query. Describe the role of the llm here
    Context: {join(documents)};                                   # This is the retrived data. Closest K docs
    Prompt: {query}                                               # This is the user query. 
    <PostPrompt>                                                  # This is the post prompt to specify additional operations on result. Describe the output format here
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

# q = f"This is regarding purchase of your new computer server. please send me purchase order of the same."
# ans = rag_pipeline.run(query = q)

# print(type(ans['results']))

# for i in ans['results']:
#     print(i.strip())

# while 1:
#     q = input("Enter query: ")
#     ans = rag_pipeline.run(query = q)
#     print(ans['results'][0])