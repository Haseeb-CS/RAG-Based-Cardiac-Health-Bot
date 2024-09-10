
RAG based Cardiac Health Chatbot



The code creates an intelligent chatbot, "Cardiac Health Bot," using Llama Index agents to provide specialized answers about cardiovascular diseases, leveraging a Retrieval-Augmented Generation (RAG) approach. It integrates OpenAI's GPT-4 with a knowledge base built from a PDF on cardiac health, ensuring responses are strictly relevant to the domain. The bot uses a memory buffer to retain conversation context, improving accuracy over time, and includes a custom tool for saving user notes, enhancing its functionality. By focusing solely on cardiac health data and avoiding unrelated content, the chatbot offers precise, context-aware assistance to users seeking information on cardiovascular disorders.


## Authors

- [@Haseeb-CS](https://github.com/Haseeb-CS)


## Features

- Domain-specific chatbot focused on cardiovascular diseases
- Uses Retrieval-Augmented Generation (RAG) for accurate information retrieval
- Integrates OpenAI's GPT-4 with a custom-built knowledge base from a cardiac health PDF
- Efficient information retrieval with VectorStoreIndex for quick and relevant responses
- Context-aware responses through a chat memory buffer that retains conversation history
- Custom function tool to save user notes, enhancing interactivity
- Strict adherence to domain-specific data, avoiding general knowledge or unrelated content
- Dynamic query handling that combines past and present user interactions for consistent accuracy
- Configurable base context to ensure the chatbot remains focused on cardiac health topics
- Real-time user interaction loop for continuous conversation support
## 🚀 About Me
I'm a Machine Learning Engineer specializing in computer vision, natural language processing (NLP), and image generation. I develop AI solutions that leverage my expertise in these domains to solve complex problems and create innovative applications.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Haseeb-CS?tab=repositories)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/shahhaseeb281)



## Installation

Follow these steps to set up and run the "Cardiac Health Bot":

Clone the Repository: Download or clone the repository to your local machine:

```
git clone https://github.com/YourUsername/Cardiac-Health-Bot.git
cd Cardiac-Health-Bot
```
Set Up a Python Environment: Ensure Python 3.x is installed on your system. It's recommended to create a virtual environment to manage dependencies:
```
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```
Install Required Libraries: Install the necessary libraries using pip:

```
pip install openai llama-index python-dotenv
```
Create a .env File:
In the root directory of your project, create a .env file and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```
Replace your_openai_api_key_here with your actual OpenAI API key.

Obtain an OpenAI API Key:

Visit the OpenAI Platform and sign up or log in.
Navigate to the API Keys section in your account settings.
Click on Create new secret key to generate a new API key.
Copy the key and paste it into your .env file as described above.
Prepare the Data: Ensure that the PDF files containing cardiac health information are placed in the data directory, as the chatbot will build its knowledge base from these documents.

Run the Code: Execute the script to start the chatbot:

```
python script_name.py
```
Interact with the Chatbot: Enter your prompts related to cardiovascular diseases, and the chatbot will provide domain-specific responses based on the PDF content.

By following these steps, you will have the "Cardiac Health Bot" set up and ready to use, providing precise and relevant answers to queries about cardiovascular diseases.