import os
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from pdf2image import convert_from_path
from PIL import Image
from docx import Document

# ==== 0️⃣ Khởi tạo LayoutLMv3 (chỉ dùng cho PDF) ====
model_name = "microsoft/layoutlmv3-base"
processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
model = LayoutLMv3Model.from_pretrained(model_name)
model.eval()

# ==============================================================
#              DOCX TEXT EXTRACTION FUNCTION 
# ==============================================================

def extract_docx_text(file_path):
    doc = Document(file_path)
    text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip() != ""])
    return text


# ==============================================================
#              PDF PROCESSING WITH LAYOUTLMv3
# ==============================================================

def extract_text_and_bboxes(pdf_path):
    words, boxes = [], []
    for page_layout in extract_pages(pdf_path):
        (page_x0, page_y0, page_x1, page_y1) = page_layout.bbox

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_text = ""
                    (line_x0, line_y0, line_x1, line_y1) = text_line.bbox

                    for character in text_line:
                        if isinstance(character, LTChar):
                            line_text += character.get_text()

                    words_in_line = line_text.strip().split()
                    if not words_in_line:
                        continue

                    x0 = int(line_x0 / page_x1 * 1000)
                    y0 = int(line_y0 / page_y1 * 1000)
                    x1 = int(line_x1 / page_x1 * 1000)
                    y1 = int(line_y1 / page_y1 * 1000)

                    for word in words_in_line:
                        words.append(word)
                        boxes.append([x0, y0, x1, y1])
    return words, boxes
    
def prepare_inputs_from_pdf(pdf_path):
    words, boxes = extract_text_and_bboxes(pdf_path)
    image = convert_from_path(pdf_path)[0].convert("RGB")

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    return encoding, words

def read_pdf_with_layout(pdf_path):
    encoding, words = prepare_inputs_from_pdf(pdf_path)
    with torch.no_grad():
        outputs = model(**encoding)
    return words, outputs.last_hidden_state


# ==============================================================
#              MASTER FUNCTION (AUTO ROUTER)
# ==============================================================

def extract_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        print("📄 Detected DOCX → Extracting using python-docx (no LayoutLMv3)")
        text = extract_docx_text(file_path)
        return {"type": "docx", "text": text}

    elif ext == ".pdf":
        print("📄 Detected PDF → Processing with LayoutLMv3")
        words, embeddings = read_pdf_with_layout(file_path)
        return {"type": "pdf", "text": " ".join(words), "words": words, "embeddings": embeddings}

    else:
        raise ValueError("❌ Unsupported file type. Only .pdf and .docx are allowed.")


# ==============================================================
#              TEST
# ==============================================================

file_path = "/kaggle/input/cv-huy/nguyen-duc-huy_1758517966_Joboko_c3e1a50bcfd6fb7f_3487225.pdf"

result = extract_file(file_path)

print("\n===== OUTPUT =====")

if result["type"] == "docx":
    print("📌 DOCX Text:")
    print(result["text"])

elif result["type"] == "pdf":
    print("📌 PDF Words Count:", len(result["words"]))
    print("📌 First words sample:", result["words"][:20])
    print("📌 Embedding shape:", result["embeddings"].shape)

import re
text = " ".join(words)

text = text.replace("\x00", "")

text = re.sub(r"\s+", " ", text).strip()
print(text)  


def jd_file_to_text(file_path):
    """
    Chuyển JD PDF/DOCX trực tiếp sang text bằng Gemini API
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    prompt = """
Bạn là trợ lý AI. Chuyển nội dung file JD PDF hoặc DOCX sau
thành văn bản sạch, loại bỏ ký tự lỗi, giữ thứ tự logic.
Trả kết quả dưới dạng text liên tục, không giải thích thêm.
"""
    
    response = genai.GenerativeModel("gemini-2.5-pro").generate_content(
        prompt=prompt,
        file=file_bytes  # Gemini sẽ tự đọc nội dung PDF/DOCX
    )
    
    clean_text = response.text.strip()
    return clean_text

file_path = "sample_job_description.pdf"  # hoặc sample_job_description.docx
jd_text = jd_file_to_text(file_path)
