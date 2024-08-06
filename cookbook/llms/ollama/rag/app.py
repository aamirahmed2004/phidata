from typing import List
import tempfile
from pathlib import Path
import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from csv_reader import CSVReader
from phi.document.reader.docx import DocxReader
from phi.document.reader.text import TextReader
# from phi.document.reader.csv import CSVReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger

from assistant import get_rag_assistant  # type: ignore

st.set_page_config(
    page_title="Local RAG",
    page_icon=":orange_heart:",
)
st.title("Local RAG Chat with PDF Demo")
st.markdown("##### :orange_heart: built using [phidata](https://github.com/phidatahq/phidata)")


def restart_assistant():
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()

def query_llm(rag_assistant: Assistant, question: str) -> str:
    with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in rag_assistant.run(question):
                response += delta  # type: ignore
                resp_container.markdown(response)
    
    return response


def main() -> None:
    # Get model
    llm_model = st.sidebar.selectbox("Select Model", options=["llama3", "llama3.1"])
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    # Get Embeddings model
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["nomic-embed-text"],
        help="When you change the embeddings model, the documents will need to be added again.",
    )
    # Set assistant_type in session state
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

    # Get the assistant
    rag_assistant: Assistant
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        logger.info(f"---*--- Creating {llm_model} Assistant ---*---")
        rag_assistant = get_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["rag_assistant"] = rag_assistant
    else:
        rag_assistant = st.session_state["rag_assistant"]

    # Create assistant run (i.e. log to database) and save run_id in session state
    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return

    # Load existing messages
    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{"role": "assistant", "content": "Upload a doc and ask me questions..."}]

    # Prompt for user input
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        response = query_llm(rag_assistant, question)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    # Load knowledge base
    if rag_assistant.knowledge_base:

        # Display number of chunks in the knowledge base
        if "num_chunks" not in st.session_state:
            st.session_state["num_chunks"] = rag_assistant.knowledge_base.vector_db.get_count()

        # -*- Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
                        st.session_state["num_chunks"] = rag_assistant.knowledge_base.vector_db.get_count()
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "Add a file (xlsx, csv, pdf, docx, txt) :page_facing_up:", type=["pdf","csv","docx","txt"], key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing file...", icon="🧠")
            rag_name = uploaded_file.name.rsplit(".",1)[0]
            extension = uploaded_file.name.rsplit(".",1)[1]
            if f"{rag_name}_uploaded" not in st.session_state:
                rag_documents = None
                if extension == "pdf":
                    reader = PDFReader()
                    rag_documents: List[Document] = reader.read(uploaded_file)
                elif extension == "csv":
                    temp_dir = Path("tempDir")
                    temp_dir.mkdir(exist_ok=True)

                    # Create a temporary directory inside tempDir
                    with tempfile.TemporaryDirectory(dir=temp_dir) as temp_path:
                        temp_file_path = temp_dir / uploaded_file.name
                        # Save the uploaded CSV file to the temporary directory
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Pass the file path to the CSVReader object
                    reader = CSVReader()
                    rag_documents: List[Document] = reader.read(temp_file_path, delimiter=",")

                elif extension == "docx":
                    reader = DocxReader()
                    temp_dir = Path("tempDir")
                    temp_dir.mkdir(exist_ok=True)

                    # Create a temporary directory inside tempDir
                    with tempfile.TemporaryDirectory(dir=temp_dir) as temp_path:
                        temp_file_path = temp_dir / uploaded_file.name
                        # Save the uploaded CSV file to the temporary directory
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Pass the file path to the CSVReader object
                    rag_documents: List[Document] = reader.read(temp_file_path)

                elif extension == "txt":
                    reader = TextReader()
                    temp_dir = Path("tempDir")
                    temp_dir.mkdir(exist_ok=True)

                    # Create a temporary directory inside tempDir
                    with tempfile.TemporaryDirectory(dir=temp_dir) as temp_path:
                        temp_file_path = temp_dir / uploaded_file.name
                        # Save the uploaded CSV file to the temporary directory
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Pass the file path to the CSVReader object
                    rag_documents: List[Document] = reader.read(temp_file_path)

                # The temporary file and directory are deleted automatically

                if rag_documents:
                    rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                    st.session_state["num_chunks"] = rag_assistant.knowledge_base.vector_db.get_count()
                else:
                    st.sidebar.error(f"Could not read .{extension} file")
                st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()

    if rag_assistant.knowledge_base and rag_assistant.knowledge_base.vector_db:
        st.sidebar.text(f"Num of chunks (for demo): \n{st.session_state['num_chunks']}")
        if st.sidebar.button("Clear Knowledge Base"):
            rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Knowledge base cleared")
            st.session_state["num_chunks"] = 0

    if rag_assistant.storage:
        rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
        new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
        if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
            logger.info(f"---*--- Loading {llm_model} run: {new_rag_assistant_run_id} ---*---")
            st.session_state["rag_assistant"] = get_rag_assistant(
                llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_rag_assistant_run_id
            )
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_assistant()

if __name__ == "__main__":
    main()
