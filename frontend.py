import streamlit as st
import requests

st.title("RAG Demo: Ask Questions over Uploaded Paper")

backend_url = st.text_input("Backend URL", "http://localhost:8000")

uploaded_file = st.file_uploader("Upload PDF or HTML", type=["pdf", "html", "htm"])
if uploaded_file:
    with st.spinner("Uploading..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        resp = requests.post(f"{backend_url}/upload", files=files)
        if resp.ok:
            st.success("Upload successful!")
        else:
            st.error(resp.text)

query = st.text_input("Ask a question about the document:")
mode = st.radio("Search mode", ["semantic", "keyword"])

if st.button("Ask") and query:
    with st.spinner("Searching..."):
        resp = requests.post(f"{backend_url}/query", json={"query": query, "mode": mode}, stream=True)
        st.subheader("Results (streamed):")
        result = ""
        for chunk in resp.iter_lines():
            if chunk:
                st.write(chunk.decode())