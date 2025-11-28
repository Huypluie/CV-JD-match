import google.generativeai as genai
import json

genai.configure(api_key="...")

prompt = f"""
Phân tích nội dung CV sau (đã loại bỏ thông tin cá nhân):

--- CV ---
{clean_text}
-------------
...
"""

response = genai.GenerativeModel("gemini-2.5-pro").generate_content(prompt)

# ✅ Trích riêng phần JSON trong kết quả
match = re.search(r"\{[\s\S]*\}", response.text)
if match:
    extracted_json = match.group(0)
    try:
        data = json.loads(extracted_json)
        print(json.dumps(data, ensure_ascii=False, indent=4))
    except:
        print("⚠️ JSON không hợp lệ:\n", extracted_json)
else:
    print("❌ Không tìm thấy JSON trong phản hồi.")
    

import os
from google import genai
from google.genai import types

# 1. Khởi tạo Client
client = genai.Client(api_key="AIzaSyDDTl24qBeuxD2tPM7N2pS8iUZfktVTQ7w")

# 2. Đọc tệp PDF dưới dạng byte
pdf_path = '/kaggle/input/cv-huy/nguyen-duc-huy_1758517966_Joboko_c3e1a50bcfd6fb7f_3487225.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

# 3. Tạo Content Part (Mime Type: application/pdf)
pdf_part = types.Part.from_bytes(
    data=pdf_bytes,
    mime_type='application/pdf'
)
jd_text = """Mô tả công việc
● Phát triển và tinh chỉnh các mô hình ngôn ngữ: Sử dụng các công cụ và framework như TensorFlow, PyTorch, và Hugging Face Transformers để xây dựng các mô hình ngôn ngữ.

● Phân tích và xử lý dữ liệu ngôn ngữ: Sử dụng các kỹ thuật NLP để phân tích, trích xuất thông tin từ văn bản, và xử lý ngôn ngữ tự nhiên.

● Thiết kế hệ thống truy xuất thông tin: Phát triển các hệ thống truy xuất thông tin từ cơ sở dữ liệu để hỗ trợ quá trình tạo ra câu trả lời chính xác và đầy đủ.

● Kết hợp truy xuất và sinh văn bản: Sử dụng các kỹ thuật RAG để kết hợp thông tin truy xuất từ các nguồn dữ liệu với khả năng sinh văn bản của mô hình.

● Nghiên cứu các kỹ thuật mới: Theo dõi và nghiên cứu các xu hướng và công nghệ mới trong lĩnh vực NLP, Chatbot và RAG.

● Tối ưu hóa hiệu suất hệ thống: Tối ưu hóa thời gian phản hồi và hiệu suất của hệ thống truy xuất thông tin.

Yêu cầu ứng viên
● Kinh nghiệm: Tối thiểu 1 năm ở vị trí tương đương.

● Trình độ học vấn: Tốt nghiệp Cao đẳng/Đại học các chuyên ngành Công nghệ Thông tin, Toán Tin, Điện tử Viễn thông, Điều khiển Tự động, hoặc các ngành liên quan.

● Kiến thức chuyên môn:

- Có hiểu biết về Machine Learning và Deep Learning.

- Kinh nghiệm làm việc với các mô hình ngôn ngữ lớn (LLM) như BERT, T5, Mistral, LLaMa, GPT, v.v.

- Có kinh nghiệm làm việc với RESTAPI, Langchain, llamaindex, ...

● Kỹ năng nghiên cứu và nền tảng:

- Khả năng nghiên cứu và áp dụng các công nghệ mới.

- Nền tảng vững chắc về cấu trúc dữ liệu và thuật toán.

Quyền lợi
● Mức lương: từ 13 - 18M/tháng (thỏa thuận khi phỏng vấn)

● Công ty đóng BHYT, BHXH, BHTN theo quy định

● Công ty cung cấp thiết bị làm việc

● Review lương 1-2 lần/năm theo năng lực

● Thưởng ngày lễ 2/9, 30/04, 1/5, ... thưởng lương tháng 13

● Thưởng kết quả kinh doanh toàn công ty cuối năm

● Du lịch, nghỉ mát hàng năm

● Môi trường làm việc năng động, chuyên nghiệp, tạo cơ hội cho nhân viên thỏa sức sáng tạo và phát triển bản thân

● Pantry: Coffee, Máy pha coffee, Tủ lạnh

Địa điểm làm việc
- Hà Nội: Khu VP tầng 3, tòa nhà CT1 Constrexim Thái Hà, Phạm Văn Đồng, Cổ Nhuế 2, Bắc Từ Liêm
"""
prompt = f"""
     lọc ra từ {clean_text}, hãy đọc và Trích xuất thông tin chuẩn chỉ trong văn bản:
        - "skills": danh sách kỹ năng(dạng list)
       - "experience_years": số năm kinh nghiệm (số nguyên)
        - "education": bằng cấp cao nhất (chuỗi)
        - "certificates": danh sách chứng chỉ có (dạng list)
    Sau đó tính matching score với JD theo từng các tiêu chí skills, experience_years, education, certificates mỗi cái gồm 3 mục:
        -"score": điểm của tiêu chí đó
        -"missing"
        -"match"
    Hãy trả lời **CHỈ** bằng một đối tượng JSON, **KHÔNG** thêm bất kỳ lời giải thích hay văn bản nào khác.
"""
# 4. Gọi API với PDF và Text Prompt
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        pdf_part, jd_text,prompt
        
         # Văn bản prompt
    ]
)

print(response.text)