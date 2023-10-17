from pathlib import Path

from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import HumanApprovalCallbackHandler



st.set_page_config(
    page_title="Contend Legal", page_icon="", layout="wide", initial_sidebar_state="collapsed"
)

"# Contend Legal"


# OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
# if not OPENAI_API_KEY:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

OPENAI_API_KEY = "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"

DB_PATH = (Path(__file__).parent / "data").absolute()
print(DB_PATH)
loader = DirectoryLoader("/workspaces/streamlit-agent/streamlit_agent/data", glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
loaded_docs = loader.load()
db = FAISS.from_documents(loaded_docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

@tool("search docs")
def search_docs(query: str) -> str:
    """Searches the database for legal information."""
    docs = db.similarity_search_with_relevance_scores(query, k=1)
    return docs[0][0].page_content

@tool("get user input")
def get_user_input(query: str) -> str:
    """Use this to ask the user a single question about their situation."""

    # Display query to user using st.write()
    # End the current st.chat_message("assistant")
    # Wait for user's response before resuming the agent
    # User's response should get added as: st.chat_message("user").write(response)
    # The agent should get resumed as a new st.chat_message("assistant")

    st.write(query)
    return "2 weeks"

tools = [search_docs, get_user_input]

model = OpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, streaming=True)
agent = initialize_agent(
    tools,
    model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True
)




# llm = OpenAI(temperature=0, streaming=True, openai_api_key=OPENAI_API_KEY)
# tools = load_tools(["ddg-search"])
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

def _approve(_input: str) -> bool:
    # msg = (
    #     "Do you approve of the following input? "
    #     "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    # )
    # msg += "\n\n" + _input + "\n"
    # resp = input(msg)
    # return resp.lower() in ("yes", "y")
    resp = input()
    st.write(resp)
    return True

# human_input_callback = HumanApprovalCallbackHandler(approve=_approve)


if prompt := st.chat_input(placeholder="Ask a legal question"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
        # response = agent.run(prompt, callbacks=[st_callback])
        # st.write(response)

        # Trying to run the agent iteratively, so that we can ask the user for input:
        st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
        for step in agent.iter(prompt, callbacks=[st_callback]):
            if output := step.get("intermediate_step"):
                action, value = output[0]
                user_query = output[0][0].tool_input

                



                # if action.tool == "get user input":
                #     _continue = st.chat_input()

                # if action.tool == "GetPrime":
                #     print(f"Checking whether {value} is prime...")
                #     assert is_prime(int(value))
                # Ask user if they want to continue
                # _continue = input("Should the agent continue (Y/n)?:\n")
                # if _continue != "Y":
                #     break
