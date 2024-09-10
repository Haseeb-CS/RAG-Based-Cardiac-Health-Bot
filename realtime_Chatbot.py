from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer, SimpleChatStore, ChatMessage
from fpdf import FPDF

load_dotenv()

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

def save_pdf_from_url(url, filename):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)

def load_new_data(sources):
    documents = []
    for source in sources:
        if source.startswith("http"):
            filename = os.path.join("data", f"website_content_{sources.index(source)}.pdf")
            save_pdf_from_url(source, filename)
            reader = SimpleDirectoryReader(input_dir="data")
        else:
            reader = SimpleDirectoryReader(input_dir=os.path.dirname(source))
        documents.extend(reader.load_data())
    
    index_name = "new_knowledge_base"
    index = get_index(documents, index_name)
    return index

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")
    with open(note_file, "a") as f:
        f.writelines([note + "\n"])
    return "note saved"

note_file = os.path.join("data", "notes.txt")
note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)

tools = [
    note_engine,
]

base_context = """Purpose: The primary role of this agent is to assist users by providing accurate factual 
            information from the PDF only. The agent must not answer any questions related to general knowledge.
            You are a CardioBot and you are trained on a specific knowledge base.
            If you do not know the answer, just say I dont know the relevant answer.
            Do not give any answer if you do not find it from the PDF.
            Do not give any answers related to general knowledge questions.
            While answering new questions, also remember past responses from {chat_memory}
            Do not provide any answer related to countries. """

# Initialize chat memory buffer
chat_memory = ChatMemoryBuffer.from_defaults(
    chat_store=SimpleChatStore(), 
    token_limit=2048
)

llm = OpenAI(model="gpt-4")
query_engine_tool = None

def create_agent(tools, llm, context, memory):
    return ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context, memory=memory)

def query_with_memory(agent, prompt):
    chat_history = chat_memory.get_all()
    context_with_memory = base_context + "\n" + "\n".join(f"User: {msg.content}" if msg.role == "user" else f"Bot: {msg.content}" for msg in chat_history)
    context_with_memory += f"\nUser: {prompt}"
    
    response = agent.query(context_with_memory)
    chat_memory.put(ChatMessage(role="user", content=prompt))
    chat_memory.put(ChatMessage(role="assistant", content=response))
    return response

def get_knowledge_base():
    sources = []
    print("Enter the knowledge base (PDF paths or URLs) one by one. Type 'done' when you are finished:")
    while True:
        source = input()
        if source.lower() == "done":
            break
        sources.append(source)
    return sources

def main():
    sources = get_knowledge_base()
    if sources:
        global query_engine_tool
        index = load_new_data(sources)
        query_engine = index.as_query_engine()
        
        if query_engine_tool:
            query_engine_tool.query_engine = query_engine
        else:
            query_engine_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="dynamic_data",
                    description="this gives detailed information from the newly added source",
                ),
            )
            tools.append(query_engine_tool)
        
        global agent
        agent = create_agent(tools, llm, base_context, chat_memory)
        print("Knowledge base uploaded and indexed successfully.")
        
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            result = query_with_memory(agent, prompt)
            print(result)
    else:
        print("No knowledge base provided. Exiting.")

if __name__ == "__main__":
    agent = create_agent(tools, llm, base_context, chat_memory)
    main()
