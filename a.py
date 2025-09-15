import os
import uuid
import streamlit as st
from textwrap import dedent
from mistralai import Mistral
from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools

# 🔑 Clé Mistral directe
mistral_key = "7SdMheI6cmDZtIb1fxFe2JkWWmLbpuI7"
client = Mistral(api_key=mistral_key)

# ===== Wrapper Mistral compatible AGNO =====
class MistralAgentModel:
    def __init__(self, api_key, model_name="mistral-7b-instruct"):
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        self.id = str(uuid.uuid4())  # requis par AGNO
        self.provider = "mistral"
        self.type = "chat"

    def run(self, prompt):
        text = prompt.content if hasattr(prompt, "content") else str(prompt)
        response = self.client.chat.create(
            model=self.model_name,
            messages=[{"role": "user", "content": text}]
        )
        return type("Response", (), {"content": response.choices[0].message["content"]})

    def to_dict(self):
        return {"id": self.id, "provider": self.provider, "type": self.type, "model_name": self.model_name}

# ===== OCR PDF =====
def ocr_pdf(pdf_path, pdf_name):
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        return

    uploaded_pdf = client.files.upload(
        file={"file_name": pdf_path, "content": open(pdf_path, "rb")},
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        include_image_base64=True
    )

    os.makedirs("Documents", exist_ok=True)
    with open(f"Documents/{pdf_name}.md", "w", encoding="utf-8") as f:
        f.write("\n".join([page.markdown for page in ocr_response.pages]))

    return True

# ===== Fonctions pour récupérer les documents =====
def export_law_document(agent, query, num_documents=None, **kwargs):
    with open("Documents/uk-export-law.md", "r", encoding="utf-8") as f:
        return f.read()

def import_law_document(agent, query, num_documents=None, **kwargs):
    with open("Documents/morocco-import-law.md", "r", encoding="utf-8") as f:
        return f.read()

# ===== Agents =====
translation_agent = Agent(
    name="Translation Agent",
    model=MistralAgentModel(api_key=mistral_key),
    instructions=dedent("""
        You are an expert translator. Translate French markdown to English while keeping formatting intact.
    """)
)

specialist_agent = Agent(
    name="Specialist Agent",
    model=MistralAgentModel(api_key=mistral_key),
    instructions=dedent("""
        You are an Export-Import Compliance Expert. Provide step-by-step UK export and Moroccan import process
        including licences, taxes, duties, permits, prohibited items, documents, compliance rules, and penalties.
    """),
    tools=[GoogleSearchTools()]
)

# ===== Streamlit UI =====
st.title("Import Export Specialist Multi-Agent System")
product_details = st.text_input("Enter your product and shipment details")

if st.button("Submit"):
    with st.spinner("Running analysis..."):
        # OCR des documents
        ocr_pdf('Documents/uk-export-law.pdf', 'uk-export-law')
        ocr_pdf('Documents/morocco-import-law.pdf', 'morocco-import-law')

        # Traduction du document marocain
        morocco_text = import_law_document(None, None, None)
        translation_response = translation_agent.run(morocco_text)
        with open("Documents/morocco-import-law.md", "w", encoding="utf-8") as f:
            f.write(translation_response.content)

        # Affecter le retriever
        specialist_agent.retriever = [export_law_document, import_law_document]

        # Exécuter l'agent spécialiste
        specialist_response = specialist_agent.run(product_details)

        # Afficher le résultat
        st.markdown(specialist_response.content)
