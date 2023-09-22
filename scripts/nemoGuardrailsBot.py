import tqdm
import pinecone
import openai
import os
from dotenv import load_dotenv
import argparse
import asyncio

from nemoguardrails import LLMRails, RailsConfig

from utilities import retrieve, rag

# Get CLI args
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--index", help="pinecone vector db index name")

cli_args = parser.parse_args()

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

openai.api_key = OPENAI_API_KEY

# Define embedding model
embed_model_id = "text-embedding-ada-002"

# Init Pinecone vector db
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(cli_args.index)

# Load rails config
config = RailsConfig.from_path("../nemo_guardrails/config_bot")
rag_rails = LLMRails(config)
rag_rails.register_action(action=retrieve, name="retrieve")
rag_rails.register_action(action=rag, name="rag")

async def main():

	res = await rag_rails.generate_async(prompt=query)
	print(res)


if __name__ == "__main__":

	while True:

		query = input("User query: ")

		if query == "exit":
			break
		else:
			loop = asyncio.get_event_loop()
			loop.run_until_complete(main())




