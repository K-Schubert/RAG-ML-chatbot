# Import libraries
import os
import argparse

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool
from langchain.agents import initialize_agent

import pinecone

from dotenv import load_dotenv

from utilities import get_metadata, augment_prompt

# Get CLI
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--semanticsearch", action=argparse.BooleanOptionalAction, help="perform semantic similarity search on knowledge base")
parser.add_argument("-q", "--qa", action=argparse.BooleanOptionalAction, help="generative RAG QA")
parser.add_argument("-c", "--chatbot", action=argparse.BooleanOptionalAction, help="converse with a RAG chatbot")
parser.add_argument("-i", "--index", help="pinecone vector db index name")
parser.add_argument("-k", "--k_res", help="k most similar documents to be returned from vector db")
parser.add_argument("-m", "--max_it", help="max reasoning/action iterations performed by chatbot")
parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, help="chatbor verbosity")

cli_args = parser.parse_args()

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

# Init Pinecone Vector DB
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

text_field = "text"
index_name = cli_args.index
index = pinecone.Index(index_name)

# Init embedding model
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002",
	                               disallowed_special=())

# Initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

if __name__ == "__main__":

	# Similarity Search
	if cli_args.semanticsearch:

		while True:

			query = input("User query: ")

			if query == "exit":
				break
			else:
				res = vectorstore.similarity_search(query, k=int(cli_args.k_res))

				[print({"title": doc.metadata["title"],
					 "document": doc.page_content,
					 "source": doc.metadata["source"]}) for doc in res]

	# RAG QA
	if cli_args.qa:
		
		llm = ChatOpenAI(
		    openai_api_key=OPENAI_API_KEY,
		    model_name='gpt-3.5-turbo',
		    temperature=0.0
		)

		qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
		    llm=llm,
		    chain_type="stuff",
		    retriever=vectorstore.as_retriever(fetch_k=20, k=int(cli_args.k_res), return_source_documents=True)
		)

		while True:

			query = input("User query: ")

			if query == "exit":
				break
			else:
				res = qa_with_sources(query)

				print(f'Query: {res["question"]}')
				print(f'Answer: {res["answer"]}')
				print(f'Sources: {res["sources"]}')

	# RAG Conversational Chatbot
	if cli_args.chatbot:
		
		llm = ChatOpenAI(
		    openai_api_key=OPENAI_API_KEY,
		    model_name='gpt-3.5-turbo',
		    temperature=0.0
		)

		# conversational memory
		conversational_memory = ConversationBufferWindowMemory(
		    memory_key='chat_history', # refers to conversational agent component
		    k=int(cli_args.k_res),
		    return_messages=True
		)

		qa = RetrievalQA.from_chain_type(
								    llm=llm,
								    chain_type="stuff",
								    retriever=vectorstore.as_retriever(fetch_k=20, k=int(cli_args.k_res), return_source_documents=True)
								)

		"""qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
								    llm=llm,
								    chain_type="stuff",
								    retriever=vectorstore.as_retriever(fetch_k=20, k=int(cli_args.k_res), return_source_documents=True)
								)"""

		# Add retrievalQA tool to agent
		tools = [
		    Tool(
		        name='Knowledge Base',
		        func=qa.run,
		        description=(
		            'use this tool when answering general knowledge queries to get '
		            'more information about the topic'
		        )
		    )
		]

		agent = initialize_agent(
		    agent='chat-conversational-react-description',
		    tools=tools,
		    llm=llm,
		    verbose=cli_args.verbose,
		    max_iterations=int(cli_args.max_it),
		    early_stopping_method='generate',
		    memory=conversational_memory
		)

		while True:

			query = input("User query: ")

			if query == "exit":
				break
			else:
				res = agent(query)
				
				print(f'Answer: {res["output"]}')
				#print(res)
				#print(f'Sources: {res["sources"]}')
