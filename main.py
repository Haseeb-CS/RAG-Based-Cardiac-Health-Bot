from dotenv import load_dotenv
import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer, SimpleChatStore, ChatMessage



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


cardiac_pdf = SimpleDirectoryReader(input_dir='data')
cardiac_documents = cardiac_pdf.load_data()
cardiac_index = get_index(cardiac_documents, "cardiac")
cardiac_engine = cardiac_index.as_query_engine()

# *****************************************

note_file = os.path.join("data", "notes.txt")


def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n"])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)


tools = [
    note_engine,
    
    QueryEngineTool(
        query_engine=cardiac_engine,
        metadata=ToolMetadata(
            name="cardiac_data",
            description="this gives detailed information about Cardiac Health Disorders from the PDF only",
        ),
    ),
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
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=base_context, memory=chat_memory)



def query_with_memory(agent, prompt):
    chat_history = chat_memory.get_all()
    context_with_memory = base_context + "\n" + "\n".join(f"User: {msg.content}" if msg.role == "user" else f"Bot: {msg.content}" for msg in chat_history)
    context_with_memory += f"\nUser: {prompt}"
    
    response = agent.query(context_with_memory)
    chat_memory.put(ChatMessage(role="user", content=prompt))
    chat_memory.put(ChatMessage(role="assistant", content=response))
    return response



while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = query_with_memory(agent, prompt)
    print(result)