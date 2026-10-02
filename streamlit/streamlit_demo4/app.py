import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
# --- Page Config ---
st.set_page_config(page_title="EduRAG: Trustworthy AI", layout="wide")
st.markdown(
    """
    <style>
    .stAppDeployButton, 
    [data-testid="stAppDeployButton"], 
    [data-testid="stToolbar"] > div:first-child {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0px !important;
        height: 0px !important;
        pointer-events: none !important;
    }

    header[data-testid="stHeader"], 
    [data-testid="stHeaderActions"], 
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        z-index: -9999 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("EduRAG: Institutional Knowledge Demo")
st.markdown("Evaluating **Hallucination vs. Traceability** across general AI and controlled RAG architectures.")


# --- Sidebar: Setup & Ingestion ---
with st.sidebar:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY not found. Please check your .env file.")

    st.header("Knowledge Base")
    uploaded_files = st.file_uploader("Upload Course Materials (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Upload Source Files", type="primary"):
        if not api_key:
            st.error("Cannot initialize database without a valid API key.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF document.")
        else:
            with st.spinner(f"Processing {len(uploaded_files)} document(s) and generating embeddings..."):

                all_docs = []

                for uploaded_file in uploaded_files:
                    temp_filepath = os.path.join(tempfile.gettempdir(), uploaded_file.name)

                    with open(temp_filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    loader = PyPDFLoader(temp_filepath)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata['source'] = uploaded_file.name

                    all_docs.extend(docs)

                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)

                os.environ["GOOGLE_API_KEY"] = api_key
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

                vectorstore = FAISS.from_documents(splits, embeddings)

                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                st.success("Database active. Ready for queries.")

tab1, tab2, tab3 = st.tabs(
    ["1. What this presentation is about?", "2. General AI vs. EduRAG", "3. Hybrid Search (Auto-Fallback)"])

with tab1:
    st.header("The core question: how this is useful for actual University (or school) use?")
    st.markdown("""
    ### The Challenge of AI Hallucinations

    As anyone who has used a standard generative AI model likely knows, these systems occasionally make mistakes or fabricate information—a phenomenon known as "hallucinating." This occurs because standard large language models (LLMs) are fundamentally designed to predict the next plausible word and provide an answer at all costs. When an AI lacks the exact factual data, it rarely admits ignorance. Instead, it generates a response based on statistically adjacent information. This results in plausible but entirely fabricated outputs that can easily mislead users.

    ### What is a RAG System?

    RAG stands for **Retrieval-Augmented Generation**. Instead of relying solely on an AI model's internal, pre-trained memory to answer a question, a RAG architecture acts as a digital researcher. 

    When a user submits a prompt, the system first *retrieves* highly relevant information from a closed, verified database of specific documents (such as uploaded PDFs, course syllabi, or policy manuals). It then provides this exact retrieved text to the AI model, instructing it to *generate* an answer based exclusively on those trusted documents. 

    ### How RAG Solves the Trust Issue

    The RAG model presented in this demonstration is explicitly constrained by strict system instructions. If the answer to a query cannot be found within the uploaded source files, the model is programmed to decline the prompt rather than guess. 

    While this means the system will not answer every conceivable question like a standard, open-ended chatbot, it guarantees **factual reliability and traceability**. Every generated response is directly anchored to your specific materials, ensuring that when the AI does provide an answer, it can be trusted entirely.

    ### Future Practical Applications

    The applications for this architecture in educational and institutional settings are vast. A primary use case is creating a secure university or school portal where professors upload verified course materials, allowing students to interact with a reliable, closed-loop AI assistant.

    A different approach to balance factual strictness with a seamless user experience, institutions can also deploy a **Hybrid Search Architecture** (as demonstrated in the third tab of this application):
    * **Primary Tier (Verified RAG):** The system first searches the institution's trusted source files. If the answer is found, it delivers a guaranteed, explicitly cited response.
    * **Secondary Tier (General AI Fallback):** If the data is absent from the local files, the system falls back to the general LLM to provide an answer. However, it distinctly flags this response with a warning that the information is drawn from outside the institution's verified database, maintaining total transparency.
    """)

with tab2:
    # --- Main UI: Query & Comparison ---
    st.header("Ask the Assistant")
    query = st.text_input("Enter a question about the course materials:")

    if query:
        if "retriever" not in st.session_state:
            st.warning("Please upload documents and initialize the database first.")
        elif not api_key:
            st.error("Cannot query model without a valid API key.")
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

            # Retrieve relevant documents for the RAG chain
            retriever = st.session_state.retriever
            retrieved_docs = retriever.invoke(query)

            context_parts = []
            for doc in retrieved_docs:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                page_num = doc.metadata.get('page', 0) + 1
                context_parts.append(f"[SOURCE: {source_name} | PAGE: {page_num}]\n{doc.page_content}")

            context_text = "\n\n".join(context_parts)

            # --- CHAIN A: General LLM ---
            general_prompt = ChatPromptTemplate.from_template("Answer the following question: {question}")
            general_chain = general_prompt | llm | StrOutputParser()

            # --- CHAIN B: EduRAG ---
            refusal_string = "I cannot answer this based on the provided institutional materials."

            rag_prompt = ChatPromptTemplate.from_template(f"""
            You are a strict institutional assistant. Answer the question based ONLY on the following context.
            If the answer is not in the context, reply EXACTLY with: "{refusal_string}"

            When you provide an answer, you MUST append a structured list of the sources you used at the very end of your response. Use the [SOURCE: X | PAGE: Y] tags provided in the context to build your citations.

            Format your response EXACTLY like this:
            [Your detailed answer here]

            ---
            **Sources Used:**
            * `[Source File Name]` | **Page [Number]**

            Context:
            {{context}}

            Question: {{question}}
            """)
            rag_chain = rag_prompt | llm | StrOutputParser()

            with st.spinner("Generating responses..."):
                general_response = general_chain.invoke({"question": query})
                rag_response = rag_chain.invoke({"context": context_text, "question": query})

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("General LLM (Prone to Hallucination)")
                st.info(general_response)
                st.caption(
                    "Notice how the model relies on its internal training data, which may be culturally biased or factually incorrect for your specific university.")

            with col2:
                st.subheader("EduRAG (Institutionally Controlled)")
                st.success(rag_response)

                with st.expander("View Retrieved Sources & Traceability"):
                    st.caption("The RAG assistant used the following exact excerpts to generate its answer:")

                    for i, doc in enumerate(retrieved_docs):
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                        page_num = doc.metadata.get('page', 'Unknown')
                        if isinstance(page_num, int):
                            page_num += 1

                        st.markdown(f"**Source {i + 1}** | `{source_file}` | **Page {page_num}**")

                        formatted_quote = "\n".join([f"> {line}" for line in doc.page_content.split('\n')])
                        st.markdown(formatted_quote)

                        if i < len(retrieved_docs) - 1:
                            st.divider()

with tab3:
    st.header("Hybrid Search: RAG with Auto-Fallback")
    st.markdown(
        "This tab demonstrates the practical application of a hybrid model. It first attempts to find the answer in your trusted documents. If it cannot, it falls back to the general AI, but clearly flags the response as unverified.")

    query_hybrid = st.text_input("Enter your question:", key="query_tab3")

    if query_hybrid:
        if "retriever" not in st.session_state:
            st.warning("Please upload documents and initialize the database in the sidebar first.")
        elif not api_key:
            st.error("Cannot query model without a valid API key.")
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

            retriever = st.session_state.retriever
            retrieved_docs = retriever.invoke(query_hybrid)

            context_parts = []
            for doc in retrieved_docs:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                page_num = doc.metadata.get('page', 0) + 1
                context_parts.append(f"[SOURCE: {source_name} | PAGE: {page_num}]\n{doc.page_content}")

            context_text = "\n\n".join(context_parts)

            refusal_string = "I cannot answer this based on the provided institutional materials."

            rag_prompt = ChatPromptTemplate.from_template(f"""
            You are a strict institutional assistant. Answer the question based ONLY on the following context.
            If the answer is not in the context, reply EXACTLY with: "{refusal_string}"

            When you provide an answer, you MUST append a structured list of the sources you used at the very end of your response. Use the [SOURCE: X | PAGE: Y] tags provided in the context to build your citations.

            Format your response EXACTLY like this:
            [Your detailed answer here]

            ---
            **Sources Used:**
            * `[Source File Name]` | **Page [Number]**

            Context:
            {{context}}

            Question: {{question}}
            """)
            rag_chain = rag_prompt | llm | StrOutputParser()

            with st.spinner("Searching institutional knowledge base..."):
                rag_response = rag_chain.invoke({"context": context_text, "question": query_hybrid})

            if refusal_string in rag_response:
                st.warning(
                    "**Institutional Data Not Found:** The following answer is generated by general AI and is NOT verified by your uploaded materials.")

                general_prompt = ChatPromptTemplate.from_template("Answer the following question: {question}")
                general_chain = general_prompt | llm | StrOutputParser()

                with st.spinner("Generating general AI response..."):
                    general_response = general_chain.invoke({"question": query_hybrid})

                st.info(general_response)

            else:
                st.success("**Verified Institutional Answer**")
                st.write(rag_response)

                with st.expander("View Retrieved Sources & Traceability"):
                    st.caption("The RAG assistant used the following exact excerpts to generate its answer:")

                    for i, doc in enumerate(retrieved_docs):
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                        page_num = doc.metadata.get('page', 'Unknown')
                        if isinstance(page_num, int):
                            page_num += 1  # 0-indexed

                        st.markdown(f"**Source {i + 1}** | `{source_file}` | **Page {page_num}**")

                        formatted_quote = "\n".join([f"> {line}" for line in doc.page_content.split('\n')])
                        st.markdown(formatted_quote)

                        if i < len(retrieved_docs) - 1:
                            st.divider()