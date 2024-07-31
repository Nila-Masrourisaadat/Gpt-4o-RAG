Project Name: Retrieval-Augmented Generation (RAG) System
Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval with generative AI to provide accurate and contextually relevant answers to user questions. The system processes documents, stores them in a vector database, and uses the OpenAI GPT-4 model to generate responses based on the retrieved information.

Features:

Document Processing: Handles PDF and Word documents, splits them into chunks, and stores them in a vector database.
Interactive Shell: Allows users to input questions and receive answers interactively.
Error Handling: Manages errors gracefully, including unsupported file types and processing issues.
Files in the Repository
rag.py: The main script that initializes the system, processes documents, and handles user interactions.
rag_llm_wrapper.py: Contains functions for interacting with the OpenAI GPT-4 API.
config.py: Configuration settings for the system.
rag_text_helper.py: Helper functions for text processing and splitting.

Inspiration:

Some of the RAG code was inspired by the OgbujiPT repository: https://github.com/OoriData/OgbujiPT

