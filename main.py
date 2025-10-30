from fastapi import FastAPI, UploadFile
from transformers import pipeline
import PyPDF2
import docx

app = FastAPI(title="Nextcloud Summarizer")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text(file: UploadFile):
    filename = file.filename.lower()
    content = ""
    if filename.endswith(".txt"):
        content = file.file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        content = "".join([page.extract_text() for page in reader.pages])
    elif filename.endswith(".docx"):
        doc = docx.Document(file.file)
        content = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type")
    return content.strip()

@app.post("/summarize")
async def summarize(file: UploadFile):
    try:
        text = extract_text(file)
        if not text:
            return {"error": "No text extracted"}
        summary = summarizer(text[:2000], max_length=150, min_length=50, do_sample=False)
        return {"summary": summary[0]["summary_text"]}
    except Exception as e:
        return {"error": str(e)}
