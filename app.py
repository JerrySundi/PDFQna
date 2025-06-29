import streamlit as st
import os
import shutil
from addPDF import process_pdfs
from qna import get_answer
from logger import logger

def clear_session_state():
    """Helper function to clear all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    logger.info("Cleared all session state variables")

def main():
    logger.info("Starting the application")
    st.title("PDF Question Answering Bot")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    uploaded_files = st.file_uploader("Choose PDF file(s)", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        # Create a temporary directory to store uploaded PDFs
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        # Save uploaded files to the temporary directory
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        # Process PDFs
        logger.info("Processing PDFs")
        process_pdfs(temp_dir)
        st.success("PDFs processed successfully!")
        logger.info("PDF/s loaded successfully")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask a question about the PDFs"):
            logger.info(f"question received successfully -> {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            logger.info("Getting API response")
            response = get_answer(prompt)
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            logger.info("Displayed AI response")

        # Show End Chat button only if there are messages
        if st.session_state.messages:
            # Create a container for the button at the bottom
            button_container = st.container()
            with button_container:
                if st.button("End Chat", key="end_chat"):
                    logger.info("Ending chat session")
                    # Delete the temporary PDF directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        logger.info(f"Deleted temporary directory: {temp_dir}")
                    
                    # Clear all session state
                    clear_session_state()
                    
                    # Force a rerun of the app
                    st.rerun()

if __name__ == "__main__":
    main()