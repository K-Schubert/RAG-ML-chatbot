import openai
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

openai.api_key = OPENAI_API_KEY

def get_metadata(query: str):

    results = vectorstore.similarity_search(query, k=3)

    metadata = [{"title": x.metadata["title"], 
                "source": x.metadata["source"]} for x in result]

    return metadata

def augment_prompt(query: str):
    
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    
    return augmented_prompt

async def retrieve(query: str) -> list:

    # init Pinecone vector db
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index("rag-ml")
    # define embedding model
    embed_model_id = "text-embedding-ada-002"
    # create query embedding
    res = openai.Embedding.create(input=[query], engine=embed_model_id)
    xq = res['data'][0]['embedding']
    # get relevant contexts from pinecone
    res = index.query(xq, top_k=5, include_metadata=True)
    # get list of retrieved texts
    contexts = [x['metadata']['text'] for x in res['matches']]
    return contexts

async def rag(query: str, contexts: list) -> str:

    print("> RAG Called")  # we'll add this so we can see when this is being used
    context_str = "\n".join(contexts)
    # place query and contexts into RAG prompt
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {context_str}

    Query: {query}

    Answer: """
    # generate answer
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=100
    )
    return res['choices'][0]['text']



