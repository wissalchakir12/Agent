import os
import glob
from textwrap import dedent
from agno.agent import Agent
from agno.models.mistral import MistralChat
from agno.vectordb.pgvector import PgVector
from agno.tools.googlesearch import GoogleSearchTools
from agno.knowledge import TextKnowledgeBase
from html2docx import html2docx
import streamlit as st

# --- Clé Mistral ---
mistral_key = os.environ.get("MISTRAL_API_KEY")

# --- Client Mistral pour OCR ---
from mistralai import Mistral
client = Mistral(api_key=mistral_key)

# --- OCR PDF ---
def ocr_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        st.error("PDF file not found")
        return

    uploaded_pdf = client.files.upload(
        file={
            "file_name": pdf_path,
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=True
    )

    md_path = f"Markdown/{os.path.basename(pdf_path).replace('.pdf','')}.md"
    os.makedirs("Markdown", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join([page.markdown for page in ocr_response.pages]))

    print(f"✅ Markdown saved to {md_path}")


# --- Décommenter pour lancer l'OCR une seule fois ---
# ocr_pdf("Documents/morocco-import-law.pdf")
# quit()


# --- Knowledge Base (TextKnowledgeBase à partir des fichiers .md) ---
texts = []
for file in glob.glob("Markdown/*.md"):
    with open(file, "r", encoding="utf-8") as f:
        texts.append(f.read())

knowledge_base = TextKnowledgeBase(
    texts=texts,
    vector_db=PgVector(
        table_name="markdown_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

knowledge_base.load(recreate=False)


# --- Import-Export Agent ---
import_export_agent = Agent(
    name='Import Export Agent',
    model=MistralChat(id="mistral-instruct-beta", api_key=mistral_key),
    instructions=dedent("""
    You are an Export-Import Compliance Expert with access to two knowledge sources:
    - The official UK export law document for electronics (including licences, restrictions, taxes, fees),
    - The official Moroccan import law document for electronics (including permits, quantity limits, import duties, VAT, prohibited items, and customs procedures).

    You also have access to GoogleSearchTools() to find up-to-date, authoritative data when it is missing or unclear.

    Your task is: given detailed product info (type, quantity, raw materials) and the shipment route (UK to Morocco), provide a full, precise, step-by-step explanation of UK export and Moroccan import processes including licenses, taxes, documentation, restrictions, duties, VAT, and customs procedures. Use only facts from your knowledge base or live sources.
    """),
    expected_output=dedent("""
    IMPORTANT: Output the final result in raw HTML, starting directly with <h1> and without wrapping it in Markdown code blocks.

    <h1 style="text-align: center;">📦 Import-Export Compliance Report</h1>
    <hr>
    <!-- UK Export Process, Moroccan Import Process, Taxes & Compliance Details, Summary -->
    """),
    knowledge=knowledge_base,
    search_knowledge=True,
    tools=[GoogleSearchTools()]
)


# --- Local Regulations Agent ---
local_regulations_agent = Agent(
    name='Local Regulations Agent',
    model=MistralChat(id="mistral-instruct-beta", api_key=mistral_key),
    instructions=dedent("""
    You are a Local Regulation Checker Agent. Research local regulations in a specific country related to the imported product. Focus on legality, labeling, certification, licenses, distribution rules, and taxes. Use only official, reliable sources.
    """),
    expected_output=dedent("""
    IMPORTANT: Output the final result in raw HTML, starting directly with <h1> and without wrapping it in Markdown code blocks.

    <h1 style="text-align: center;">📦 Local Regulation Report</h1>
    <hr>
    <!-- Product Legality, Labeling & Standards, Certification, Licenses, Distribution, Taxes, Sources -->
    """),
    tools=[GoogleSearchTools()]
)


# --- Run Agents ---
output_html = ""

agent1 = import_export_agent.run("An electronic air purifier exported from the UK and imported into Morocco")
output_html += agent1.content
output_html += "<hr>"

agent2 = local_regulations_agent.run("An electronic air purifier exported from the UK and imported into Morocco")
output_html += agent2.content


# --- Convert HTML to DOCX ---
docx_bytes = html2docx(output_html, title="Local Regulation Report")
with open("regulation_report.docx", "wb") as f:
    f.write(docx_bytes.getvalue())

print("✅ Saved to regulation_report.docx")
