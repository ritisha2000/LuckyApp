import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import spacy
from collections import defaultdict

st.title("Automating stuff")

pdf_file = st.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    reader = PdfReader(pdf_file)
    page = reader.pages[0]
    text = page.extract_text()

    summarizer = pipeline("summarization")
    summary = summarizer(text)

    st.header("Preview of the text")
    st.write(text[:500])

    st.header("Summary of the text")
    st.write(summary[0]["summary_text"])

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)

    st.header("People metioned")
    for person in entities["PERSON"]:
        st.markdown("- " + person)

    st.header("Organizations mentioned")
    for org in entities["ORG"]:
        st.markdown("- " + org)
