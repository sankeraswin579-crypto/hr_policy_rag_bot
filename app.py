import streamlit as st

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text

from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="HR Policy Bot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 HR Policy Q&A Bot")

st.write(
    "Upload HR PDFs and ask questions."
)

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_text = ""

    with st.spinner("Reading PDFs..."):

        for pdf in uploaded_files:

            all_text += load_pdf(pdf)

    chunks = split_text(all_text)

    st.success(
        f"Created {len(chunks)} chunks"
    )

    rag = RAGPipeline()

    with st.spinner(
        "Building FAISS index..."
    ):

        rag.build_vectorstore(
            chunks
        )

    st.success(
        "Vector Database Ready"
    )

    question = st.text_input(
        "Ask a policy question"
    )

    if question:

        with st.spinner(
            "Searching documents..."
        ):

            answer, sources = rag.answer(
                question
            )

        st.subheader("Answer")

        st.write(answer)

        with st.expander(
            "Retrieved Context"
        ):

            for i, source in enumerate(
                sources,
                start=1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(source)