# RAG-ML
A repository to experiment with a specialized AI Assistant Chatbot for ML Research.

The field of AI is moving fast and it sometimes feels difficult to keep up with the latest research. Indeed, new papers come out at an increasingly fast pace, and some state-of-the-art results are ancient history no less than a month after their release. Benchmark results are being broken daily, and keeping up with the field can perhaps be a little overwhelming.

This Chatbot tool allows researchers or ML enthusiasts to be kept updated about the latest developments in the field of AI in general using the latest Chatbot technology. Indeed, one can simply query this Chatbot and ask questions about the latest benchmark results for a given task, or obtain explanations about some niche ML technique based on cutting-edge research papers, with sources. One can also use the Chatbot to select references for writing a research paper or creating a basis for a related works section.

This project aims to create a Database of ML research papers in NLP (Natural Language Processing), AI (Artificial Intelligence), ML (Machine Learning), CV (Computer Vision) and MA (Multiagent Systems). Papers are scraped from Arxiv, then embedded into a Pinecone Vector Database Index to be used as context into a RAG (Retrieval Augmented Generation) Chatbot System. The system can then produce answers to queries based on relevant context. Such a grounded system reduces LLM hallucination, provides relevant up-to-date answers with sources and is able to answer "I don't know" if the provided context is not sufficient.

Additionally, I implement RAG Agents that have the ability to search the internet for answers and extend the context knowledge to SERP results (blogs, news articles, etc.) as well as established knowledge bases (eg. Wikipedia). I also implement NeMo Guardrails to add programmable guardrails to LLM-based conversational systems.

Several RAG implementation are tested in this project:

- Naive RAG: A Semantic Similarity Search is performed for every user query (slower).
- RAG Agent: An AI Agent decides when to use a Vector DB Similarity Search depending on the query (faster).
- NeMo Guardrails:
- FLARE RAG:
- RETRO: ??

The user can then query the Chatbot to retrieve specialized up-to-date content and provide answers with sources on specific ML topics.

### To Do

#### Arxiv Scraping
- [x] Setup Arxiv bot
- [x] Setup Asyncio Web scraping
- [x] Create master list of Arxiv papers
- [x] Remove duplicated papers

#### PDF text extraction
- [x] Parse PDFs to text with PyPDF2
- [x] Base content/references extraction
- [] Setup multithreading for PDF parsing
- [] LLM content/references extraction
- [] Nougat markdown extraction

#### Text preprocessing
- [x] Text tiling (chunking) with RecursiveCharacterTextSplitter
- [] Try different text tiling methods

#### ConversationalMemory
- [] Setup Chatbot ConversationalMemory

#### Pinecone Vector DB
- [x] Setup Pinecone Vector DB
- [x] Insert document embeddings into Vector DB.

Use "dotproduct" as similarity metric for text-embedding-ada-002 model

#### RAG Chatbot
- [x] Setup Naive RAG Chatbot
- [] Setup RAG Agent Chatbot
- [] Setup RAG Guardrails Chatbot
- [] Setup RAG FLARE Chatbot

#### Agent Tools
- [] Add Calculator tool
- [] Add InternetSearch tool
- [] Add chart captioning tool

#### Open-Source Implementation
- [] Setup LLama2
- [] Setup sentence-transformers for document embeddings
- [] Tokenizer
- [] FAISS

#### LLM Fine-Tuning
- [] Fine-Tune LLM on document corpus with QLoRA
- [] Setup Pytorch DDP training

#### Performance Evaluation
- [] Assess quality of RAG outputs




## Ethical Considerations
- RAG allows to ground LLM with facts, reduce hallucinations, be more truthful
- Customize chatbot to given group of users
- Open-Source LLM to control data, privacy, usage
- Provides sources for fact-checking
- Using Arxiv papers which are not yet peer reviewed

- Constituional AI

## Arxiv Scraping

## Data Preprocessing
### Text Tiling
- tiktoken(cl100k_base)

## Naive RAG
- Using OpenAI API gpt-3.5-turbo
- Using ada-002-embeddings
- ConversationMemoryBuffer

- When to query KB?
	- Can setup a similarity threshold if a retrieved context is below threshold don't include in context
	- Use retrieval tool with Agent

## ReAct RAG Agent
Agents use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning to the user.
Use LLM as a reasoning engine. Non-deterministic sequence of actions.
Connect LLM to other sources of data/knowledge: search, APIs, DBs, calculators, run code, etc.

ReAct (Reasoning Acting): 
	- Challenge 1: Using tools in appropriate scenarios
	instructions, tool descriptions in prompt, tool retrieval, few-shot examples, fine tuned model (toolformer)
	- Challenge 2: Not using tools when not needed
	- Challenge 3: parsing LLM output to tool invocation (output parsers)
	- Challenge 4: Memory of previous reasoning steps (n most recent actions/observations combined with k most actions/observations)
	- Memory: remembering user-ai interactions, ai-tool interactions
	- Challenge 5: incorporate long observations (parse long output, store long output and do retrieval on it for next steps, eg. from API call)
	- Challengee 6: agent stays focused (reiterate objective, separate planning/execution step)
	- Evaluation: evaluate end result, intermediate steps (correct action, action input, sequence of steps, most efficient sequence of steps)

Other types of Agents:
	- AutoGPT: different objective than ReAct (initial goals for autogpt are open ended goals such as increase twitter following VS ReAct is short lived quantifiable goals) -> autogpt has long-term memory in agent-tool interactions through vectorstore
	- BabyAGI: long term memory of agent-tool interactions, has separate planning/execution steps
	- CAMEL: 2 agents in a simulation environment (chatroom), simulation good for evaluation
	- Generative Agents: 25 agents in "sims" simulated world, time/importance/relevancy-weighted memory, reflection step
	- HuggingGPT: task planner, connects AI models to solve AI tasks (ChatGPT selects models based on their huggingface description, executes subtasks and summarizes response)

## Guardrails RAG

## FLARE RAG

## Open-Source Implementation

## LLM Fine-Tuning
- Using QLoRA (mixed-bit quantization LoRA training)
- Toolformer
- Knowledge Distillation

## Chatbot
## Conversational Memory
By default, Chains and Agents are stateless, meaning that they treat each incoming query independently. In some applications (chatbots being a GREAT example) it is highly important to remember previous interactions, both at a short term but also at a long term level. The concept of “Memory” exists to do exactly that.
--> chatbot considers previous interactions

--> use LLM to learn facts about user, create user profile, store information that can be retrieved

## Performance Evaluation
Compare LLMS: OpenAI ChatGPT, Base LLama2 7b, Llama2 7b fine-tuned (4-bit, 8-bit quantization) (all with RAG)
Quantization of Llama2
Fine-Tuning: QLoRA, Batch size, gradient accumulation steps, gradient checkpointing, etc.
RLHF, RLAIF, ReST
--> fine-tune llama2 7b makes it as accurate as larger foundation model and faster to run

Evaluate Memory modules
Evaluate Embedding models: ada-002, sentence-transformers


### ConversationBufferMemory:
 the conversation buffer memory keeps the previous pieces of conversation completely unmodified, in their raw form.

	- + Store maximum information
	- - Store all tokens -> slower response time and higher cost
	- - With gpt-3.5-turbo, once we hit 4096 input tokens the model cannot process queries

### ConversationSummaryMemory
the conversation summary memory keeps the previous pieces of conversation in a summarized form, where the summarization is performed by an LLM. Our final history is shorter. This will enable us to have many more interactions before we reach our prompt's max length, making our chatbot more robust to longer conversations.
	
	- + Less token usage for long conversations
	- + Enables longer conversations
	- - Inefficient for shorter conversations
	- - Dependant on good summaries

### ConversationBufferWindowMemory
Another great option for these cases is the ConversationBufferWindowMemory where we will be keeping a few of the last interactions in our memory but we will intentionally drop the oldest ones - short-term memory if you'd like. Here the aggregate token count and the per-call token count will drop noticeably. We will control this window with the k parameter. The conversation buffer window memory keeps the latest pieces of the conversation in raw form


	- + Less token usage
	- - Chatbot can forget previous interactions if k is set too low.

### ConversationSummaryBufferMemory
the conversation summary memory keeps a summary of the earliest pieces of conversation while retaining a raw recollection of the latest interactions.

	- + Define number of past tokens to keep in memory (keep raw information about recent interactions)
	- + Summarization of past interactions (remembers distant interactions)
	- - Increased token count for shorter conversations
	- - Summary quality is not guaranteed
	- - Summary + buffer uses more tokens

### ConversationKnowledgeGraphMemory
It is based on the concept of a knowledge graph which recognizes different entities and connects them in pairs with a predicate resulting in (subject, predicate, object) triplets. This enables us to compress a lot of information into highly significant snippets that can be fed into the model as context. The conversation knowledge graph memory keeps a knowledge graph of all the entities that have been mentioned in the interactions together with their semantic relationships.

### ConversationEntityMemory
The conversation entity memory keeps a recollection of the main entities that have been mentioned, together with their specific attributes.

The final choice of conversational memory depends on the use case and predicted number of interactions. It is possible to combine different memory modules in the same chain.

## How to use
```
# create and activate virtualenv
python3 -m venv venv_rag
source venv_rag/bin/activate
```

Use the following CLI args:
	- ```s```: semantic search
	- ```q```: question answering
	- ```c```: chatbot
	- ```i```: pinecone index name
	- ```k```: k most similar docs to return from vector db
	- ```m```: maximum reasoning/actions steps to take
	- ```v```: verbosity of agent output

### Semantic Search
```
python3 naiveRAGbot.py -s -i rag-ml -k 2
```

![qa](media/ss.png)

### RAG Agent QA
```
python3 naiveRAGbot.py -q -i rag-ml -k 3
```
![qa](media/qa.png)

### RAG Agent Chatbot
```
python3 naiveRAGbot.py -c -i rag-ml -k 3 -m 3
```

![Chatbot interaction](media/chatbot.png)
