import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from typing import Union
from langchain_core.messages import SystemMessage

# Load environment variables from .env file
load_dotenv()

if "processed_files" not in st.session_state: st.session_state.processed_files = set()
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_response" not in st.session_state:
    st.session_state.agent_response = ""

def format_chat_history(memory_variables):
    chat_history = memory_variables.get("chat_history", [])
    formatted_history = []
    for message in chat_history[-4:]:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"Assistant: {message.content}")
    return "\n".join(formatted_history)



def build_prompt_string(user_input: str, chat_history_text: str) -> str:
    return f"""
You are a helpful assistant.

Chat History:
{chat_history_text}

User Question:
{user_input}

Respond using available tools.
""".strip()


async def run_mcp_agent(user_message: str) -> str:
    server_params = StdioServerParameters(command="python", args=["server.py"])
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_react_agent(model, tools)

            # 🔧 Use system message to instruct the agent
            system_message = SystemMessage(content="""
You are a helpful assistant that can use tools to query data and search indexed documents.
If unsure, always try using a tool. Only guess if absolutely necessary.
""")

            # 🧠 Get memory messages from chat_memory
            history = st.session_state.chat_memory.load_memory_variables({})["chat_history"]

            # 👤 Current prompt
            human_message = HumanMessage(content=user_message)

            # 🧠 Run agent with full context
            response = await agent.ainvoke({
                "messages": [system_message] + history + [human_message]
            })

            return response["messages"][-1].content





# --- Streamlit UI ---
st.set_page_config(page_title="MCP Agent", layout="centered")
st.title("📧 MCP Agent Assistant")
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

user_prompt = st.chat_input("You are a helpful assistant...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    chat_history_text = "\n".join([
        f"User: {m['content']}" if m['role'] == "user" else f"Assistant: {m['content']}"
        for m in st.session_state.messages
    ])
    with st.spinner("Thinking via MCP agent..."):
        try:
            response_text = asyncio.run(run_mcp_agent(f"{user_prompt} \n\n(Use indexed documents and tools if needed.)"))
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.agent_response = response_text
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})


    # --- RIGHT PANEL: Document Management and Email/PDF Form in Sidebar ---
with st.sidebar:
    st.subheader("📄 Document Management")

    if "oracle_status" not in st.session_state:
        with st.spinner("Checking Oracle DB connection..."):
            st.session_state.oracle_status = asyncio.run(run_mcp_agent("Check Oracle DB connection"))

    st.info(f"Oracle Vector Store: {st.session_state.oracle_status}")
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_id = f"{file.name}_{file.size}"
            if file_id not in st.session_state.processed_files:
                st.session_state.processed_files.add(file_id)
                with st.spinner(f"Storing {file.name}..."):
                    file_path = f"/tmp/{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                        response = asyncio.run(run_mcp_agent(f"Please index this document: {file_path}"))
                        st.success(response)

# --- Display chat messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.agent_response:
    st.markdown("---")
    st.markdown("### 💡 Last Response")
    st.markdown(st.session_state.agent_response)
