# RAG-ML
A repository to experiment with a specialized AI Assistant chatbot for ML Research.

This project aims to create a Database of ML research papers in NLP (Natural Language Processing), AI (Artificial Intelligence), ML (Machine Learning), CV (Computer Vision) and MA (Multiagent Systems). Papers are scraped from Arxiv, then embedded into a Pinecone Vector Database Index to be used as context into a RAG Chatbot System. The system can then produce answers to queries based on relevant context. Such a grounded system reduces LLM hallucination, provides relevant up-to-date answers and is able to answer "I don't know" if the provided context is not sufficient.

Several RAG implementation are tested in this project:

- Naive RAG: A Semantic Similarity Search is performed for every user query (slower).
- RAG Agent: An AI Agent decides when to use a Vector DB Similarity Search depending on the query (faster).
- Guardrails RAG:
- FLARE RAG:

The user can then query the Chatbot to retrieve specialized up-to-date content or answers on specific ML topics.

## Naive RAG

## RAG Agent

## Guardrails RAG

## FLARE RAG

## Arxiv Scraping

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

#### Pinecone Vector DB
- [x] Setup Pinecone Vector DB
- [x] Insert document embeddings into Vector DB.

#### Text preprocessing
- [] Text tiling (chunking)

#### RAG Chatbot
- [x] Setup Naive RAG Chatbot
- [] Setup RAG Agent Chatbot
- [] Setup RAG Guardrails Chatbot
- [] Setup RAG FLARE Chatbot

## How to use
