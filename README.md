# Indian Constitution Chatbot

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Features](#features)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Common Issues](#common-issues)
- [License](#license)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)
- [References](#references)

## Project Overview

The **Indian Constitution Chatbot** is an AI-powered chatbot that answers questions related to the Indian Constitution in 2024. It is built using the **Retrieval-Augmented Generation (RAG)** method, utilizing cutting-edge embeddings and models from **Hugging Face** to retrieve relevant data and generate informative responses.

This chatbot aims to provide users with an easy way to understand the Indian Constitution's articles, clauses, and more.

## Installation

### Prerequisites
- Python 3.x
- Docker (for deployment on Hugging Face)
- faiss-cpu
- PyPDF2
- langchain_google_genai
- langchain
- streamlit
- langchain_community
- python-dotenv
- fastapi
- google-generativeai
- langchain_huggingface

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/mhdnihas/Indian-Constitution-Bot.git

2. Navigate to the project directory:
   ```bash
   cd Indian-Constitution-Bot

3. Set up a virtual environment:
   ```bash
   python3 -m venv chatbot_env

4. Activate the virtual environment:

    * On Windows:
      ```bash
      chatbot_env\Scripts\activate  

    * On macOS/Linux
      ```bash
      source chatbot_env/bin/activate

5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

6. Set up environment variables (if applicable):

   * Create a .env file in the project root and add your API keys or other environment-specific variables.

7. Run the backend:
   * Use Uvicorn to start the FastAPI server:
   ```bash
   uvicorn app:app --reload

8. Open the web interface in your browser:

    * Go to http://127.0.0.1:8000 to access the chatbot.



## Usage

To interact with the **Indian Constitution Chatbot**, you can use the following options:

### Live Link

You can access the chatbot directly via the live link provided below:

- [Indian Constitution Chatbot (Live)](https://nihas2218-Indian-Constitution-Bot.hf.space)

This link will take you to the deployed version of the chatbot, where you can start asking questions about the Indian Constitution. The chatbot will provide answers based on the latest information available in the Constitution of India.

### API Endpoint

For developers looking to integrate the chatbot into other systems or applications, you can interact with the backend using the following API endpoint:

- **API Endpoint:** [https://nihas2218-Indian-Constitution-Bot.hf.space/chatbot](https://nihas2218-Indian-Constitution-Bot.hf.space/chatbot)

#### To make requests to the chatbot:

- Use a **POST** request to the endpoint.
- Send the query in the request body.
- The chatbot will return an appropriate response based on the Indian Constitution.

#### Example request:

```bash
curl -X POST https://nihas2218-Indian-Constitution-Bot.hf.space/chatbot \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Article 370 of the Indian Constitution?"}'


## How it Works:

The **Indian Constitution Chatbot** uses the **Retrieval-Augmented Generation (RAG)** technique to answer questions about the Indian Constitution. It combines powerful natural language processing (NLP) models and a vector database for efficient retrieval of relevant information. Here's a breakdown of how the system works:

1. **Initialization of Models and Data**:
   - On application startup, the models (`llm` and `embeddings`) are loaded, and the relevant metadata is fetched.
   - The documents related to the Indian Constitution are loaded, processed, and stored in a vector database (FAISS). The vector database stores the embeddings of the Constitution's sections and articles for efficient retrieval.

2. **Document Creation and Vectorstore Configuration**:
   - The `configure_vectorstore` function creates a **FAISS vectorstore** using the Constitution documents and the pre-trained embeddings.
   - The vectorstore is saved locally for future use, improving the chatbot's performance by making document retrieval faster.

3. **User Query Handling**:
   - When a user asks a question, the system processes the input and checks for relevant information in the stored vector database.
   - The `generate_response_with_retrieval_chain` function is used to generate a response to the user's query by retrieving relevant context and then using a **retrieval chain** to generate a response.
   - The response is based on the context retrieved from the database and the model's capabilities.

4. **Retrieval and Response Generation**:
   - A **ChatPromptTemplate** is used to format the prompt sent to the language model, which provides a structure for the response.
   - The retriever searches for relevant documents in the vectorstore using the `k`-nearest neighbor approach (retrieving the top 7 most relevant sections).
   - The model then generates a response based on the relevant context, providing an informative answer to the user's question.

5. **API and Frontend**:
   - The backend uses **FastAPI** to handle incoming requests, with an endpoint (`/chatbot`) that listens for user queries.
   - The chatbot interface is served as an HTML page, allowing users to interact with the chatbot through a simple UI.
   - The `respond_to_chat_request` function processes the user message, generates the appropriate response, and returns it as a JSON object.

6. **Error Handling**:
   - If there are issues with generating the response or retrieving the necessary data, the application catches exceptions and returns a meaningful error message.

### Key Components:
- **Vectorstore (FAISS)**: Stores the embeddings of Constitution-related documents to enable quick retrieval.
- **Retrieval Chain**: Combines the search results from the vectorstore and the language model to generate a final response.
- **Prompt Template**: Defines how the chatbot should respond to different types of user queries.
- **FastAPI**: Provides the backend infrastructure to handle user requests via API endpoints.

This architecture enables the chatbot to answer detailed and contextually relevant questions about the Indian Constitution by utilizing both retrieval-based and generation-based methods.



## Features

The **Indian Constitution Chatbot** provides the following features:

1. **Answering Questions on the Indian Constitution**:
   - The chatbot can answer a wide range of questions related to the Indian Constitution, including articles, amendments, and fundamental rights.
   - It leverages the latest information available in the Indian Constitution of 2024 to provide accurate and reliable responses.

2. **Retrieval-Augmented Generation (RAG)**:
   - The chatbot utilizes a **Retrieval-Augmented Generation (RAG)** method, which combines the power of **retrieval-based models** (FAISS vectorstore) with **generative models** to provide contextually accurate answers.
   - The system retrieves relevant sections from the Constitution and uses this information to generate human-like, detailed answers.

3. **Fast and Efficient Information Retrieval**:
   - The chatbot stores pre-processed Constitution documents in a **FAISS vectorstore**, enabling fast and efficient retrieval of relevant sections.
   - Using **sentence-transformers**, the chatbot can quickly match user queries to the most relevant information.

4. **Contextual and Informative Responses**:
   - The chatbot generates responses based on both the context retrieved from the vectorstore and the knowledge embedded in the model, ensuring informative answers to specific queries.
   - The model is trained to provide detailed answers for specific sections or general information about the Indian Constitution.

5. **User-Friendly Interface**:
   - The chatbot is deployed with a simple web-based interface that allows users to easily interact and ask questions about the Constitution.
   - It supports text-based queries and provides quick responses, making it accessible and easy to use for all users.

6. **API Endpoint for Integration**:
   - The chatbot provides an API endpoint (`/chatbot`) for developers who want to integrate the chatbot into other systems or applications.
   - The API supports **POST** requests, where developers can send user queries and receive a response based on the Constitution.

7. **Error Handling and Robust Responses**:
   - The chatbot includes error handling to manage unexpected inputs or issues during the query process.
   - If the chatbot cannot retrieve the relevant information, it provides helpful messages like, "Sorry, I couldn't find an answer to your question."

8. **Flexible Model Deployment**:
   - The chatbot is deployed on **Hugging Face**, making it easily accessible through a live link and API endpoint.
   - Docker is used for packaging and deploying the chatbot, ensuring consistency and ease of use in different environments.

9. **Comprehensive Coverage**:
   - The chatbot covers all key aspects of the Indian Constitution, including articles, fundamental rights, duties, amendments, and more.
   - Users can inquire about specific articles or general topics related to the Constitution.

10. **Customizable for Future Use**:
    - The chatbot is designed to be extensible, allowing for additional features or enhancements, such as adding more documents, expanding the coverage of specific articles, or supporting multiple languages in the future.

These features make the **Indian Constitution Chatbot** a versatile tool for anyone looking to learn more about the Constitution or integrate it into their applications.



## Deployment

The **Indian Constitution Chatbot** is deployed on **Hugging Face Spaces** using **Docker** to ensure portability and consistency across different environments. The deployment process involves building a Docker container that encapsulates the necessary dependencies, model files, and API server, and then hosting the app on Hugging Face for easy access.

### Deployment Steps

1. **Set Up Docker**:
   - The chatbot is packaged into a Docker container to ensure that the application runs consistently across various platforms. The `Dockerfile` contains all the necessary instructions to install dependencies, set up the environment, and run the chatbot.
   - The Docker container includes the FastAPI app, model files, vectorstore, and other dependencies required for the chatbot to function.

2. **Building the Docker Image**:
   To build the Docker image locally (for testing or development), follow these steps:
   ```bash
   docker build -t indian-constitution-chatbot .



## API Documentation

The **Indian Constitution Chatbot** exposes a simple API that allows users to interact with the system programmatically. This API is built using **FastAPI** and is hosted on **Hugging Face Spaces**. It allows developers to send POST requests with user queries and receive responses based on the Indian Constitution.

### Base URL

- **Base URL**: [https://nihas2218-Indian-Constitution-Bot.hf.space](https://nihas2218-Indian-Constitution-Bot.hf.space)
- **API Endpoint**: [https://nihas2218-Indian-Constitution-Bot.hf.space/chatbot](https://nihas2218-Indian-Constitution-Bot.hf.space/chatbot)

### Request Format

The API expects a **POST** request with a JSON body containing the user's query.

#### Request Body
```json
{
  "message": "What is Article 370 of the Indian Constitution?"
}


## Contact Information

For any questions, suggestions, or issues regarding the **Indian Constitution Chatbot** project, feel free to reach out.

- **Author**: Muhammed Nihas
- **Email**: [muhammednihas2218@gmail.com](muhammednihas2218@gmail.com)
- **GitHub**: [https://github.com/mhdnihas](https://github.com/mhdnihas)
- **LinkedIn**: [https://www.linkedin.com/in/muhammed-nihas-2a8a18260/](https://www.linkedin.com/in/muhammed-nihas-2a8a18260/)

You can also open an issue on the repository if you encounter any bugs or need help with the project. Contributions and suggestions are welcome!



## Acknowledgments

- **Hugging Face**: For providing the platform and tools to deploy and host the chatbot using their Spaces and API.
- **FastAPI**: For enabling easy and efficient backend API development for handling chatbot requests.
- **FAISS**: For providing a high-performance vector search library, which was used for efficient document retrieval in the chatbot.
- **Langchain**: For their powerful tools to help with the creation of the retrieval and document chains, enabling seamless integration of language models.
- **Sentence-Transformers**: For providing the `all-MiniLM-L6-v2` model, which helped with generating embeddings for document retrieval.
- **Docker**: For simplifying the deployment process by containerizing the application and ensuring consistency across environments.
- **Open Source Community**: For the countless open-source contributions and tools that made this project possible.

I would like to thank everyone who contributed to the open-source ecosystem, as well as any individuals who have offered support, feedback, and suggestions.


## Contributing

We welcome contributions to improve the **Indian Constitution Chatbot**! If you are interested in contributing, here’s how you can get started:

### How to Contribute

1. **Fork the Repository**: Click the "Fork" button at the top right of this repository to create a copy of the project in your own GitHub account.
2. **Clone the Repository**: Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/your-username/Indian-Constitution-Bot.git



## Future Improvements

While the **Indian Constitution Chatbot** is functional, there are several areas for enhancement and future development. Some of the potential improvements include:

### 1. **Expansion of Knowledge Base**
   - **Adding More Articles**: The chatbot can be expanded to include a more comprehensive set of articles, amendments, and legal information.
   - **Multilingual Support**: Implement support for additional languages to make the chatbot accessible to a broader audience.
   - **Dynamic Knowledge Update**: Incorporate a mechanism to automatically update the chatbot’s knowledge base whenever new amendments or constitutional changes occur.

### 2. **Improved Natural Language Understanding (NLU)**
   - **Contextual Understanding**: Enhance the chatbot’s ability to handle more complex queries and provide context-aware responses.
   - **Fine-Tuning**: Use fine-tuning on specific legal datasets to improve accuracy in answering legal-related queries.
   - **Sentiment Analysis**: Incorporate sentiment analysis to gauge user emotions and provide a more tailored response.

### 3. **User Interaction Features**
   - **Voice Integration**: Add voice interaction capabilities, allowing users to ask questions using voice commands.
   - **Visual Responses**: Provide visual representations of the Constitution, such as charts or infographics, to complement textual responses.
   - **Interactive Q&A**: Allow users to click through a Q&A interface with suggestions for further questions based on previous queries.

### 4. **Performance Enhancements**
   - **Faster Response Times**: Optimize the document retrieval process to reduce the response time for queries.
   - **Scalability**: Improve the backend infrastructure to handle increased user load efficiently, especially during peak traffic.

### 5. **Integration with External Resources**
   - **Integration with Legal Databases**: Link the chatbot with external legal databases for up-to-date information on case law, judicial interpretations, and more.
   - **Mobile App Integration**: Develop a mobile application for easier access to the chatbot.

### 6. **User Feedback System**
   - **Ratings & Feedback**: Add a feature to collect user feedback and ratings to improve the chatbot's performance over time.
   - **User History**: Implement a feature that allows users to track their history of questions and answers for future reference.

These improvements will help enhance the functionality, accessibility, and user experience of the chatbot, making it a more robust tool for learning and understanding the Indian Constitution.


## References

The following resources were utilized or inspired the development of the **Indian Constitution Chatbot**:

### 1. **Hugging Face Transformers and Embeddings**
   - [Hugging Face Documentation](https://huggingface.co/docs)
   - [Hugging Face Embeddings](https://huggingface.co/docs/transformers/main_classes/embeddings)

### 2. **Retrieval-Augmented Generation (RAG)**
   - [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

### 3. **FAISS (Facebook AI Similarity Search)**
   - [FAISS Documentation](https://github.com/facebookresearch/faiss)
   - [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/abs/1702.08734)

### 4. **LangChain**
   - [LangChain Documentation](https://langchain.readthedocs.io/en/latest/)
   - [LangChain: A Framework for Building NLP Applications](https://www.langchain.com/)

### 5. **FastAPI**
   - [FastAPI Documentation](https://fastapi.tiangolo.com/)
   - [FastAPI: Modern Web Framework for Fast APIs](https://fastapi.tiangolo.com/)

### 6. **Sentence Transformers**
   - [Sentence-Transformers Documentation](https://www.sbert.net/)
   - [Sentence Transformers: Using BERT and Friends for Similarity](https://arxiv.org/abs/1908.10084)

### 7. **Docker**
   - [Docker Documentation](https://docs.docker.com/)
   - [Docker: The Container Platform](https://www.docker.com/what-docker-is)

### 8. **Python Environment Management**
   - [Python Environment Setup](https://realpython.com/python-environments/)
   - [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)

### 9. **Other Resources**
   - [Indian Constitution - Full Text](https://legislative.gov.in/constitution-of-india)
   - [Constitution of India - Wikipedia](https://en.wikipedia.org/wiki/Constitution_of_India)
   
### 10. **OpenAI**
   - [OpenAI API Documentation](https://beta.openai.com/docs/)

These resources and frameworks provided the foundation for the chatbot’s design, functionality, and deployment.







