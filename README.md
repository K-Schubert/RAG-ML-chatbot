# RAG-ML
## Project Description
A repository to experiment with a specialized AI Assistant Chatbot for ML Research.

The field of AI is moving fast and it can be difficult to keep up with the latest research. Indeed, new papers come out at an increasingly fast pace, and some state-of-the-art results are ancient history no less than a month after their release. Benchmark results are being broken daily, and keeping up with the field can perhaps feel a little overwhelming.

This Chatbot tool allows researchers or ML enthusiasts to be kept updated about the latest developments in the field of AI in general using the latest Chatbot technology. Indeed, one can simply query the Chatbot and ask questions about the latest benchmark results for a given task, or obtain explanations about some niche ML technique based on cutting-edge research papers, with sources. One can also use the Chatbot to select references for writing a research paper or creating a "related works" section.

As a start, this project aims to create a Database of ML research papers in NLP (Natural Language Processing), AI (Artificial Intelligence), ML (Machine Learning), CV (Computer Vision) and MA (Multiagent Systems). Papers are scraped from Arxiv, then embedded into a Pinecone Vector Database Index to be used as context into a RAG (Retrieval Augmented Generation) Chatbot System. The system can then produce answers to queries based on relevant context. Such a grounded system reduces LLM hallucination, provides relevant up-to-date answers with sources and is able to answer "I don't know" if the provided context is not sufficient.

Additionally, I implement a simple semantic search functionality and a RAG Agent that has the ability to search the Arxiv Datastore and the internet for answers and extend the context knowledge to SERP results (blogs, news articles, etc.) as well as established alternative Knowledge Bases (eg. Wikidata). I also implement a NeMo Guardrails Agent to add programmable guardrails to LLM-based conversational systems, allowing the Chatbot system to decide when and when not to use RAG to speed up generation. The user can then query the Chatbot to retrieve specialized up-to-date content and provide answers with sources on specific ML topics.

At the moment, the following RAG models are implemented in this project:

- __Semantic Search__: Simple Semantic Similarity Search based on a user query.
- __RAG QA__: Question/Answering with RAG.
- __Naive RAG Agent__: An AI Agent Chatbot decides when to use a Vector DB Similarity Search depending on the query (faster).
- __NeMo Guardrails RAG Agent__: An AI Agent is programmed with guardrails to define desirable behaviour when greeting the user, acceptable chat topics as well as canonical forms and utterances to speed up generation and provide relevant context from the Datastore/Knowledge Base.

## Related Works
I present below the latest work on Retrieval Augmented Generation and promising techniques which will be implemented in this repo. The following papers follow the idea that powerful retrievers allow the use of smaller LLMs for generation, reducing training cost and latency while maintaining performance compared to large LLMs.

#### Fine-tuning/Eval on open-domain QA
These techniques mainly insert retrieved context into the LLM prompt.

	* __REALM (2020)__: Retrieve chunks of context, feed into prompt, perform one retrieval step. MLM + QA fine-tuning. [paper](https://arxiv.org/abs/2002.08909), [repo](https://github.com/google-research/language/blob/master/language/realm/README.md)
	* __DPR (2020)__: Pipeline training, fine-tuning on QA (no MLM). [paper](https://arxiv.org/abs/2004.04906), [repo](https://github.com/facebookresearch/DPR)
	* __RAG (2020)__: Generative approach instead of MLM, fine-tuning on QA and knowledge intensive tasks. [paper](https://arxiv.org/abs/2005.11401)
	* __Atlas (2022)__: Combien RAG with retrieval-based language model pre-training (encoder-decoder architecture), fine-tuning on QA and QA tasks. Analysis of efficient fine-tuning to specific domain. [paper](https://arxiv.org/pdf/2208.03299.pdf), [repo](https://github.com/facebookresearch/atlas) 

#### In-Context Retrieval (LM perplexity)
These techniques mainly use the LLM logit perplexity for the next token prediction as a signal for retrieval, insert context into prompt and retrieve-in-context.

	* __In-Context RALM (2023)__ + __REPLUG (2023)__: Shorter prefix (more recent tokens) helps and retrieving more frequently helps (but is inefficient and costly). Retrieve chunks of context, feed into prompt, perform retrieval every n tokens (n>1). [RALM paper](https://arxiv.org/abs/2302.00083), [RALM repo](https://github.com/AI21Labs/in-context-ralm), [REPLUG paper](https://arxiv.org/abs/2301.12652)
	* __RETRO (2021)__: Feed retrieved tokens (embeddings) into intermediate LLM layers (decoder head). Designed for many chunks, frequently, more efficiently. Chunked cross-attention (CCA) design: can use many blocks more frequently/efficiently BUT need training of model to learn CCA params. Increasing model parameters improves performance, the larger datastore the better (tested with 1.8 trillion tokens). Retrieve chunks of context, feed into intermediate layers, perform retrieval every n tokens (n>1). [paper](https://arxiv.org/abs/2112.04426), [repo](https://github.com/lucidrains/RETRO-pytorch)
	* __kNN-LM (2020)__: LM outputs a nonparametric distribution over tokens in data, can be seen as an incorporation in the output layer. Which tokens in datastore are close to the next token? Which prefixes in datastore are close to the current prefix? Which vectors in datastore are close to the prefix (prompt) embedding? Outperforms no-retrieval LM, better with bigger datastore, better with bigger k. Retrieve tokens, feed into output layer, perform retrieval every token. More fine-grained retrieval, can be better at rare patterns + out of domain, datastore is expensive in space vs chunks. [paper](https://arxiv.org/pdf/1911.00172.pdf), [repo](https://github.com/urvashik/knnlm)

#### Adaptive retrieval: follows retrieve-in-context (text chunks) and kNN-LM (tokens)

	* __FLARE__: Model starts answering a question, if probability of next token is low (model is uncertain), look up relevant documents, continue generation with retrieval, repeat until finished. More efficient, decision may not always be optimal. [paper](https://arxiv.org/abs/2305.06983), [repo](https://github.com/jzbjyb/FLARE)

#### Entity Retrieval
These techniques retrieve entities or entity mentions, retrieved context goes to intermediate layers, retrieval for every entity mention.

	* __Entities as Experts (2020)__: [paper](https://arxiv.org/abs/2004.07202)
	* __Mention Memory (2022)__: [paper](https://arxiv.org/abs/2110.06176)
	--> effective for entity-centric tasks and is space-efficient, but requires entity detection.

#### Retrieval for Long-Range LM

	* __Memorizing Transformers (2022)__: Datastore is based on input instead of external text corpus, kNN search incorporated in the attention layer. [paper](https://arxiv.org/abs/2203.08913), [repo](https://github.com/lucidrains/memorizing-transformers-pytorch)
	* __Unlimiformer (2023)__: [paper](https://arxiv.org/abs/2305.01625), [repo](https://github.com/abertsch72/unlimiformer)
	* __Long-Range LM with Self-Retrieval (2023)__: [paper](https://arxiv.org/abs/2306.13421). Retrieve text chunks from the input, retrieved context goes to intermediate layers, retrieval once or every n tokens.

#### Summary of Recent Developments
	* In general, more frequent retrieval equates with better performance but is slower.
	* Simplest option: add context to prompt and feed into input layer (can be slower).
	* Intermediate layer context injection is more efficient but requires model training.
	* Retrieve text chunks: datastore can be space-efficient, requires more computation.
	* Retrieve tokens: more fine-grained, compute-efficient, datastore can be space-expensive.
	* Adaptive retrieval can improve efficiency.
	* Retrieve on entities or entity mentions instead of tokens/chunks.

#### Other Insights

	* Finetuning a retrieval model + LM (small models) jointly has much better performance at test time compared to only fine tuning the LM. However it is expensive and not always possible to pretrain a LLM. Instead, use a specific architecture proposed above: [paper](https://arxiv.org/abs/2208.03299)
	* Context generation with a LLM: [paper](https://arxiv.org/abs/2209.10063) find good results, [paper](https://arxiv.org/abs/2212.10511) find bad results
	* BM25 is still good for zero-shot retrieval or out-of-domain cases: [paper](https://arxiv.org/abs/2104.08663)
	* Multilingual retriever: Contriever [paper](https://arxiv.org/abs/2112.09118), [repo](https://github.com/facebookresearch/contriever), MIA [repo](https://github.com/mia-workshop/MIA-Shared-Task-2022)
	* Huggingface RAG with Ray (2x inference speedup): [code](https://huggingface.co/blog/ray-rag)
	* Sparse retrieval with BM25 as retriever: [paper](https://arxiv.org/pdf/2212.10511.pdf), [paper](https://arxiv.org/pdf/2302.00083.pdf)
	* Prompt decomposition: [paper](https://arxiv.org/abs/2212.14024), [paper](https://arxiv.org/abs/2210.03350)
	* DSPy: [paper](https://arxiv.org/abs/2212.14024), [repo](https://github.com/stanfordnlp/dspy)
	* HyDE: [paper](https://arxiv.org/abs/2212.10496)

### To Do

#### Arxiv Scraping
- [x] Setup Arxiv bot
- [x] Setup Asyncio Web scraping
- [x] Create master list of Arxiv papers
- [x] Remove duplicated papers

#### PDF text extraction
- [x] Parse PDFs to text with PyPDF2
- [x] Base content/references extraction
- [x] Nougat markdown extraction
- [ ] LLM content/references extraction

#### Synthetic QA Dataset Creation
- [x] Create synthetic QA dataset with gpt-3.5-turbo

#### Text preprocessing
- [x] Text tiling (chunking) with RecursiveCharacterTextSplitter
- [ ] Optimize text tiling methods

#### ConversationalMemory
- [x] Setup Chatbot ConversationalMemory module
- [ ] Setup Chatbot CombinedMemory module

#### Document Retriever
- [ ] Setup ColBERT retriever

#### Pinecone Vector DB
- [x] Setup Pinecone Vector DB
- [x] Upsert document embeddings into Vector DB.
- [ ] Add paper submission date to docs for filtering

#### RAG Chatbot
- [x] Setup Naive RAG Chatbot
- [x] Setup RAG Agent Chatbot
- [x] Setup RAG Guardrails (Nvidia) Chatbot
- [ ] Setup RAG FLARE Chatbot

#### Agent Tools
- [x] Add Datastore RAG Search tool
- [ ] Add InternetSearch tool
- [ ] Add chart captioning tool

#### Open-Source Implementation
- [ ] Setup Constitutional model in Langchain
- [ ] Setup LLama2 as LLM
- [ ] Setup sentence-transformers for document embeddings
- [ ] Tokenizer
- [ ] FAISS

#### LLM Fine-Tuning
- [x] Fine-Tune LLama2 (single-gpu accelerate) on ML document corpus with 4-bit QLoRA
- [x] Fine-Tune LLama2 (multi-gpu accelerate) on ML document corpus with 4-bit QLoRA
- [ ] Fine-Tune Llama2 on ML document QA dataset
- [ ] Fine-Tune Llama2 on chatbot interaction history
- [ ] Fine-Tune Llama2 on Alpaca dataset
- [ ] Setup Pytorch DDP training

#### Performance Evaluation
- [ ] Assess quality of RAG outputs with Ragas
- [ ] Assess quality of RAG outputs with LangSmith

#### GUI
- [ ] Setup streamlit GUI


## Ethical Considerations

	* RAG allows to ground LLM with facts, reduce hallucinations, be more truthful and avoid dangerous topics
	* RAG allows Chatbot customization to a given group of users
	* Open-Source LLM and local setup to control data, privacy, usage
	* RAG provides sources for fact-checking
	* Arxiv papers are not yet peer reviewed
	* LLM can leak training data via adversarial attacks
	* Constituional AI + RLAIF: get a model to aign itself, critique itself based on constitution rules (eg. don't be harmful, unethical, toxic, sexist, dangerous, racist, illegal, etc.) through prompting and finetuning

## Arxiv Scraping
ML research papers are scraped from Arxiv in PDF format.

## Data Preprocessing

#### PDF Extraction
The first version of the project extracted PDF content using PyPDF2. However, the quality of the extracted content was somewhat dubious at times. In a second step, I used Meta's ```nougat``` model ([paper](https://arxiv.org/abs/2308.13418), [repo](https://github.com/facebookresearch/nougat)) which is built for academic paper PDF extraction. The PDFs are extracted to markdown ```.mmd``` format then converted to ```.txt``` format. The high quality extraction allows to obtain excellent quality content and references for each paper compared to PyPDF2. I then use an LLM to structure the output to "content" and "references".

#### Text Tiling

	* tiktoken(cl100k_base)

## Naive RAG

	* Using OpenAI API gpt-3.5-turbo
	* Using ada-002-embeddings
	* ConversationMemoryBuffer

	* When to query KB?
		- Can setup a similarity threshold if a retrieved context is below threshold don't include in context
		- Use retrieval tool with Agent

## ReAct RAG Agent
ReAct (Reasoning/Action) Agents use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning a response to the user. Here, the LLM is used as a reasoning engine and is connected to other sources of data/knowledge: search, APIs, DBs, calculators, run code, etc.

![qa](media/react.png)

#### ReAct (Reasoning Acting): 
	- Challenge 1: Using tools in appropriate scenarios
	instructions, tool descriptions in prompt, tool retrieval, few-shot examples, fine tuned model (toolformer)
	- Challenge 2: Not using tools when not needed
	- Challenge 3: parsing LLM output to tool invocation (output parsers)
	- Challenge 4: Memory of previous reasoning steps (n most recent actions/observations combined with k most actions/observations)
	- Memory: remembering user-ai interactions, ai-tool interactions
	- Challenge 5: incorporate long observations (parse long output, store long output and do retrieval on it for next steps, eg. from API call)
	- Challengee 6: agent stays focused (reiterate objective, separate planning/execution step)
	- Evaluation: evaluate end result, intermediate steps (correct action, action input, sequence of steps, most efficient sequence of steps)

#### Other types of Agents:
	- AutoGPT: different objective than ReAct (initial goals for autogpt are open ended goals such as increase twitter following VS ReAct is short lived quantifiable goals) -> autogpt has long-term memory in agent-tool interactions through vectorstore
	- BabyAGI: long term memory of agent-tool interactions, has separate planning/execution steps
	- CAMEL: 2 agents in a simulation environment (chatroom), simulation good for evaluation
	- Generative Agents: 25 agents in "sims" simulated world, time/importance/relevancy-weighted memory, reflection step
	- HuggingGPT: task planner, connects AI models to solve AI tasks (ChatGPT selects models based on their huggingface description, executes subtasks and summarizes response)

## Guardrails RAG
Guardrails are used for safety and topic guidance, to setup a deterministic dialogue flow, for RAG and conversational agents. Guardrails can enhance safety and reduce chatbot response latency if using RAG (Agent chooses to use a retrieval KB tool if necessary, otherwise the chatbot answers regular queries without retrieval, hence one less LLM call).

topics.co: define hardcoded behaviours (not necessary)

How does it work? Canonical forms are defined (eg. "define user ask weapon") and utterances are hardcoded (eg. "how to make a dirty bomb") and embedded into vector space. User queries are also embedded into vector space. If the query is similar to utterances of a given canonical form, the user query activates a flow according to the closest canonical form (eg. "what is PEFT?" -> "user ask LLM training"). Here I use the NeMO framework by Nvidia which defines ```topics.co``` and ```config.yaml``` files and the colang programming language to guide chatbot conversational flows.

--> finetune LLM to generate topics.co canonical forms and utterances based on ML research papers.

```
# need to export OpenAI API key and configure guardrails bot to use Nvidia CLI
export OPENAI_API_KEY={OPENAI_API_KEY}
source ~/.zshrc
nemoguardrails chat --config=config/
```

You might run into certificate problems on an Apple Silicon machine. If so run the command below:
```
bash /Applications/Python*/Install\ Certificates.command
```
## FLARE RAG

## Open-Source Implementation
After prototype a base system with OpenAI ```gpt-3.5-turbo``` API, the goal is to implement an open-source RAG Chatbot, leveraging ```Llama-2``` as the LLM and sentence-transformers as the vector embedding model. 

#### Vector Embedding Fine-Tuning
A model such as ```text-embedding-ada-002``` trained on classic Information Retrieval datasets (eg. MSMarco, TREC, BeIR, MTEB) may not adapt well to out of domain text. It is thus of interest to investigate the performance of a custom embedding model trained on a specialized corpus.
--> Use fine-tuned LLM to generate synthetic training data.

### LLM Fine-Tuning

#### QLoRA
It is necessary to fine-tune the LLM to boost out-of-the-box performance. Experiments have shown that LLama-2 65B trained with [QLoRA](https://arxiv.org/abs/2305.14314) on the alpaca dataset achieves 99.3% of the performance level of ChatGPT. QLoRA (Quantization-Aware training) is a paradigm leveraging PEFT (LoRA) and quantization to train a model on a single GPU or multiple GPUs with limited memory. LoRA modifies the linear projection layers of self-attention blocks using low rank matrices which reduces number of trainable parameters while preserving the model parametric knowledge. The LLM weights are frozen quantized to 4-bit and only the LoRA adapter weights are finetuned in 16-bit. Other innovations such as the introdution of the ```nf4``` datatype, double quantization and page optimizers allow me to efficiently train ```LLama-2-7b``` in parallel on 4 Tesla-V100 GPUs in XX hours. For more details on the multi-GPU training implementation, see the [qlora-multi-gpu repo](https://github.com/ChrisHayduk/qlora-multi-gpu/tree/main) and [huggingface](https://huggingface.co/blog/pytorch-ddp-accelerate-transformers).

#### Causal Language Modelling (CLM)
The first fine-tuning round of ```LLama-2-7b``` consists of optimizing the loss on the CLM task (autoregressive method where the model is trained to predict the next token in a sequence given the previous token) with a corpus of ML documents (about 60'000, only the content section). Here I use 4-bit QLoRA training on 4 GPUs.

```
# with python3 (model parallelism, device_map="balanced)
python3 scripts/train_ddp.py
```

```
# with accelerate
accelerate config
accelerate launch --config_file /home/kieran/.cache/huggingface/accelerate/default_config.yaml --main_process_port 29501 scripts/train_ddp.py
```
#### QA finetuning
Here I finetune ```llama-2-7b``` on the scraped ML papers dataset, where I generate a question using OpenAI ```gpt-3.5-turbo``` based on a chunk of text from the papers. The goal is to produce a ML QA dataset for finetuning.

#### Conversation finetuning
Here I finetune ```llama-2-7b``` on a chatbot conversation dataset.

#### Alpaca finetuning
Here I finetune ```llama-2-7b``` on the Alpaca dataset (format: task instruction, input context, response) in the self-instruct style for model alignment:

```
# alpaca example
"Below is an instruction that describes a task, paired with an input that provides further context. "
"Write a response that appropriately completes the request.\n\n"
"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
```

#### Toolformer finetuning
Here I finetune an LLM with [toolformer](https://arxiv.org/abs/2302.04761) using this [repo](https://github.com/lucidrains/toolformer-pytorch), to improve tool usage by the LLM.

## Latency Optimization
I try out several techniques to improve model inference latency while preserving generation quality.

#### Vector Database Search
Try out FAISS, other optimisations for vector search (inverted index, approximate kNN, Trees, LSH, vector quantization, etc.)

#### Model Architecture
Try out different proposed methods described in [Related Works](#related-works).

#### Knowledge Distillation
Here I distill the finetuned LLM with [Knowledge Distillation](https://arxiv.org/abs/1503.02531). The idea is to train a small student model which learns to align its outputs on a larger teacher model, achieving similar performance at a much lower cost. Aligning the student model on the output probability distribution of the teacher model provides much richer information to the student compared to training on the training data (eg. only a one-hot encoded label for a classification task).

	* + Smaller model, big speed gain at inference
	* - Need access or train a teacher model, expensive: requires ~5-10% of teacher training compute (DistilBert is 40% smaller, 60% faster and retains 97% of NLU capabilities of BERT, but trained on 8 16GB V100s for 90 hours)

#### Post-Training Model Quantization
Here I quantize the finetuned distilled LLM using [GPTQ](https://arxiv.org/pdf/2210.17323.pdf)/GGML with the [huggingface implementation](https://huggingface.co/docs/transformers/main_classes/quantization).

	* Weight quantization: store model weights in ```int8``` and dequantize to ```fp32``` for inference. This is not faster but saves space (fp32->int8: 4x).
	* Activation quantization: convert all inputs/outputs into ```int8``` and do computations in ```int8```. Need for (static/dynamic) calibration to determine layer scale factors. This allows faster inference but can require specific hardware.

#### Post-Training Model Pruning
The idea is to remove some connections in the neural network, resulting in a sparse network. The choice of pruning method depends on the inference hardware.

	* Magnitude pruning: select pruning factor x (proportion of connections to remove), set lower x% of weights (by absolute value) to 0. Sometimes retrain the model for a few more epochs after pruning. Inference is not faster and doesn't save disk space. However, using sparse matrix multiplication allows faster inference if the hardware supports these operations.
	* Structured pruning: follow a pruning structure. Store only non-0 values according to a structured sparsity pattern (compression).

#### Flash-Attention
I will implement [Flash-Attention](https://arxiv.org/abs/2205.14135) using this [repo](https://github.com/Dao-AILab/flash-attention). Attention has O(n^2) time and memory complexity. Traditionally, K, Q and V vectors are stored in High Bandwidth Memory (HBM) which is large in memory but slow in processing. Traditional attention implementation loads keys, queries, and values from HBM to GPU on-chip SRAM, performs a single step of the attention mechanism, writes it back to HBM, and repeats this for every single attention step. Flash-Attention loads the K, Q and V vectors once, fuses the operations of attention and writes them back to memory.

![flash-attention](media/flash_attention.png)

#### Other
I will implement [Multi-Query Attetion (MQA)](https://arxiv.org/abs/1911.02150), [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245), [Rotary embeddings](https://arxiv.org/abs/2104.09864), [Alibi](https://arxiv.org/abs/2108.12409) and DeepSpeed to achieve lower latency on generation. I will also explore multiprocessing for inputs and model/data parallelism for inference.

	* ONNX inference library: 
	* Tensorflow Light library:
	* MQA:
	* GQA:
	* Rotary Positional Embeddings (RoPE): Apply rotation to token vector instead of adding positional embedding by m*theta according to position m of token in the sentence. Relative positions of tokens are preserved. (ALiBi~Sinusoïdal>rotary>T5 relative).
	* ALiBi:
	* Smart K/V caching: don't recompute matmul between K.V at each generation step, takes up a lot of memory (2*precision_bytes*n_layers*d_model*seqlen*batch). Once KV cache is computed, lower latency per token generation.
	* DeepSpeed: weights are moved to and from CPU to GPU depending on the required layers in the forward pass.

![qa](media/inference_optim.png)

## Chatbot Implementation

#### Conversational Memory
By default, Chains and Agents are stateless, meaning that they treat each incoming query independently. In some applications, it is highly important to remember previous interactions, both at a short term but also at a long term level. The concept of “Memory” exists to do exactly that. The final choice of conversational memory depends on the use case and predicted number of interactions. It is possible to combine different memory modules in the same chain.

--> use LLM to learn facts about user, create user profile, store information that can be retrieved

#### ConversationBufferMemory:
 The conversation buffer memory keeps the previous pieces of conversation completely unmodified, in their raw form.

	* + Store maximum information
	* - Store all tokens -> slower response time and higher cost
	* - With gpt-3.5-turbo, once we hit 4096 input tokens the model cannot process queries

#### ConversationSummaryMemory
The conversation summary memory keeps the previous pieces of conversation in a summarized form, where the summarization is performed by an LLM. The final history is shorter. This will enable us to have many more interactions before we reach our prompt's max length, making our chatbot more robust to longer conversations.
	
	* + Less token usage for long conversations
	* + Enables longer conversations
	* - Inefficient for shorter conversations
	* - Dependant on good summaries

#### ConversationBufferWindowMemory
The ConversationBufferWindowMemory keeps a few of the last interactions in memory but intentionally drops the oldest ones. The window size is controlled with the k parameter. The conversation buffer window memory keeps the latest pieces of the conversation in raw form

	* + Less token usage
	* - Chatbot can forget previous interactions if k is set too low.

#### ConversationSummaryBufferMemory
The conversation summary memory keeps a summary of the earliest pieces of conversation while retaining a raw recollection of the latest interactions.

	* + Define number of past tokens to keep in memory (keep raw information about recent interactions)
	* + Summarization of past interactions (remembers distant interactions)
	* - Increased token count for shorter conversations
	* - Summary quality is not guaranteed
	* - Summary + buffer uses more tokens

#### ConversationKnowledgeGraphMemory
It is based on the concept of a knowledge graph which recognizes different entities and connects them in pairs with a predicate resulting in (subject, predicate, object) triplets. This enables us to compress a lot of information into highly significant snippets that can be fed into the model as context. The conversation knowledge graph memory keeps a knowledge graph of all the entities that have been mentioned in the interactions together with their semantic relationships.

#### ConversationEntityMemory
The conversation entity memory keeps a recollection of the main entities that have been mentioned, together with their specific attributes.

## Performance Evaluation

	* Compare LLMS: OpenAI ChatGPT, Base LLama2 7b, Llama2 7b fine-tuned (4-bit quantization) (all with RAG)
	* Quantization of Llama2
	* Fine-Tuning: QLoRA, Batch size, gradient accumulation steps, gradient checkpointing, etc.
	* RLHF, RLAIF, ReST
	--> fine-tuning llama2 7b makes it as accurate as larger foundation model and faster to run

	* IR: retriever (eg. ColBERT) with ONNX: single-digit ms latency on B of tokens

	* Evaluate Memory modules: long, mid, short term
	* Evaluate Embedding models: ada-002, sentence-transformers
	* Evaluate context size

	* Use langsmith, ragas

	* IR datasets: TREC, MSMarco, BEIR
	* Recall@k, Precision@k, human eval
	* Engagement metrics
	* Multi-objective ranking
	* Query distribution (tail queries)

	* --> model retrieves exact match in relevant documents or ranks documents by relevancy

	* 1. Evaluate if context is sufficient to answer question
	* 2. Evaluate generation quality (hallucination)

	* Evaluate LM with perplexity (-log(logit)) or downstream task accuracy

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

### RAG guardrails Agent Chatbot
```
python3 nemoGuardrailsBot.py -i rag-ml
```
![qa](media/guardrails.png)