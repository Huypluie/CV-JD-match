# PARSER PDF LAYOUT NHIỀU CỘT - PHÁT HIỆN VÀ SẮP XẾP ĐÚNG THỨ TỰ

"""
Giải pháp cho PDF có layout 2-3 cột:
1. Phát hiện vị trí các text blocks
2. Phân loại thành các cột
3. Đọc theo thứ tự: cột trái → cột phải → xuống dòng
"""

import fitz  # PyMuPDF
import re
from collections import defaultdict

# ============================================================================
# GIẢI PHÁP 1: PYMUPDF VỚI PHÁT HIỆN CỘT TỰ ĐỘNG
# ============================================================================

def extract_pdf_with_column_detection(pdf_path, column_count=2):
    """
    Trích xuất PDF với phát hiện cột tự động
    
    Args:
        pdf_path: Đường dẫn file PDF
        column_count: Số cột (2 hoặc 3), hoặc 'auto' để tự động phát hiện
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num, page in enumerate(doc, 1):
        # Lấy kích thước trang
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Lấy text blocks với vị trí
        blocks = page.get_text("dict")["blocks"]
        
        # Phân loại blocks theo cột
        text_blocks = []
        for block in blocks:
            if "lines" in block:  # Text block
                x0, y0, x1, y1 = block["bbox"]
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "x_center": (x0 + x1) / 2,
                        "y_center": (y0 + y1) / 2
                    })
        
        # Tự động phát hiện số cột nếu column_count == 'auto'
        if column_count == 'auto':
            column_count = detect_column_count(text_blocks, page_width)
        
        # Sắp xếp blocks theo cột và vị trí
        if column_count == 2:
            sorted_text = sort_two_column_layout(text_blocks, page_width)
        elif column_count == 3:
            sorted_text = sort_three_column_layout(text_blocks, page_width)
        else:
            # Single column - chỉ cần sort theo y
            text_blocks.sort(key=lambda b: b["y0"])
            sorted_text = "\n".join([b["text"] for b in text_blocks])
        
        full_text.append(f"--- Trang {page_num} ---\n{sorted_text}")
    
    doc.close()
    return "\n\n".join(full_text)


def detect_column_count(blocks, page_width):
    """
    Tự động phát hiện số cột dựa trên phân bố x của các blocks
    """
    if not blocks:
        return 1
    
    x_centers = [b["x_center"] for b in blocks]
    
    # Chia page thành 2 nửa
    left_blocks = sum(1 for x in x_centers if x < page_width / 2)
    right_blocks = sum(1 for x in x_centers if x >= page_width / 2)
    
    # Nếu cả 2 nửa đều có nhiều blocks → 2 cột
    if left_blocks > 3 and right_blocks > 3:
        return 2
    
    return 1


def sort_two_column_layout(blocks, page_width):
    """
    Sắp xếp text blocks cho layout 2 cột
    Đọc theo thứ tự: trái→phải, trên→dưới
    """
    # Chia thành 2 cột dựa trên x_center
    left_column = []
    right_column = []
    
    mid_x = page_width / 2
    
    for block in blocks:
        if block["x_center"] < mid_x:
            left_column.append(block)
        else:
            right_column.append(block)
    
    # Sắp xếp mỗi cột theo vị trí y (từ trên xuống)
    left_column.sort(key=lambda b: b["y0"])
    right_column.sort(key=lambda b: b["y0"])
    
    # Ghép text theo thứ tự: cột trái trước, cột phải sau
    result = []
    
    # Cột trái
    if left_column:
        result.append("=== CỘT TRÁI ===")
        for block in left_column:
            result.append(block["text"])
    
    # Cột phải
    if right_column:
        result.append("\n=== CỘT PHẢI ===")
        for block in right_column:
            result.append(block["text"])
    
    return "\n".join(result)


def sort_three_column_layout(blocks, page_width):
    """
    Sắp xếp text blocks cho layout 3 cột
    """
    left_column = []
    middle_column = []
    right_column = []
    
    third_x = page_width / 3
    
    for block in blocks:
        x = block["x_center"]
        if x < third_x:
            left_column.append(block)
        elif x < 2 * third_x:
            middle_column.append(block)
        else:
            right_column.append(block)
    
    # Sắp xếp theo y
    left_column.sort(key=lambda b: b["y0"])
    middle_column.sort(key=lambda b: b["y0"])
    right_column.sort(key=lambda b: b["y0"])
    
    result = []
    for col_name, column in [("TRÁI", left_column), ("GIỮA", middle_column), ("PHẢI", right_column)]:
        if column:
            result.append(f"\n=== CỘT {col_name} ===")
            result.extend([b["text"] for b in column])
    
    return "\n".join(result)


# ============================================================================
# GIẢI PHÁP 2: SẮP XẾP THÔNG MINH HỚN - THEO VÙNG SEMANTIC
# ============================================================================

def extract_pdf_smart_layout(pdf_path):
    """
    Trích xuất PDF với phát hiện layout thông minh
    Tự động nhận biết: header, sidebar, main content
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num, page in enumerate(doc, 1):
        page_width = page.rect.width
        page_height = page.rect.height
        
        blocks = page.get_text("dict")["blocks"]
        
        # Phân loại blocks
        text_blocks = []
        for block in blocks:
            if "lines" in block:
                x0, y0, x1, y1 = block["bbox"]
                block_text = ""
                font_sizes = []
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                        font_sizes.append(span["size"])
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "avg_font_size": sum(font_sizes) / len(font_sizes) if font_sizes else 0
                    })
        
        # Phân loại semantic
        header_blocks = []  # y < 15% page
        left_sidebar = []   # x < 30% page, y > 15%
        main_content = []   # x > 30% page, y > 15%
        
        for block in text_blocks:
            # Header (phần trên cùng)
            if block["y0"] < page_height * 0.15:
                header_blocks.append(block)
            # Left sidebar (cột trái)
            elif block["x0"] < page_width * 0.35:
                left_sidebar.append(block)
            # Main content (cột phải)
            else:
                main_content.append(block)
        
        # Sắp xếp mỗi vùng
        header_blocks.sort(key=lambda b: b["y0"])
        left_sidebar.sort(key=lambda b: b["y0"])
        main_content.sort(key=lambda b: b["y0"])
        
        # Ghép lại theo thứ tự logic
        result = []
        
        # Header (tên, title)
        if header_blocks:
            result.append("=== HEADER ===")
            for block in header_blocks:
                result.append(block["text"])
        
        # Main content trước (thường là phần quan trọng)
        if main_content:
            result.append("\n=== NỘI DUNG CHÍNH ===")
            for block in main_content:
                result.append(block["text"])
        
        # Left sidebar sau (contact info, skills)
        if left_sidebar:
            result.append("\n=== THÔNG TIN BÊN ===")
            for block in left_sidebar:
                result.append(block["text"])
        
        full_text.append(f"--- Trang {page_num} ---\n" + "\n".join(result))
    
    doc.close()
    return "\n\n".join(full_text)


# ============================================================================
# GIẢI PHÁP 3: SỬ DỤNG PDFPLUMBER VỚI CUSTOM LAYOUT
# ============================================================================

import pdfplumber

def extract_pdf_pdfplumber_columns(pdf_path, column_count=2):
    """
    Sử dụng pdfplumber để xử lý layout nhiều cột
    """
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Lấy kích thước page
            width = page.width
            height = page.height
            
            # Chia page thành các cột
            if column_count == 2:
                # Cột trái
                left_bbox = (0, 0, width / 2, height)
                left_text = page.within_bbox(left_bbox).extract_text()
                
                # Cột phải
                right_bbox = (width / 2, 0, width, height)
                right_text = page.within_bbox(right_bbox).extract_text()
                
                result = []
                if left_text:
                    result.append("=== CỘT TRÁI ===")
                    result.append(left_text)
                if right_text:
                    result.append("\n=== CỘT PHẢI ===")
                    result.append(right_text)
                
                full_text.append(f"--- Trang {page_num} ---\n" + "\n".join(result))
            
            elif column_count == 3:
                third = width / 3
                
                left_text = page.within_bbox((0, 0, third, height)).extract_text()
                middle_text = page.within_bbox((third, 0, 2*third, height)).extract_text()
                right_text = page.within_bbox((2*third, 0, width, height)).extract_text()
                
                result = []
                for name, text in [("TRÁI", left_text), ("GIỮA", middle_text), ("PHẢI", right_text)]:
                    if text:
                        result.append(f"\n=== CỘT {name} ===")
                        result.append(text)
                
                full_text.append(f"--- Trang {page_num} ---\n" + "\n".join(result))
    
    return "\n\n".join(full_text)


# ============================================================================
# GIẢI PHÁP TỐI ƯU: KẾT HỢP VÀ LÀM SẠCH
# ============================================================================

def parse_cv_with_column_detection(pdf_path, method='smart', keep_original=True):
    """
    Parse CV với phát hiện cột tự động và làm sạch
    
    Args:
        pdf_path: Đường dẫn file PDF
        method: 'auto', 'smart', 'pdfplumber'
        keep_original: Nếu True, giữ nguyên text khi không phát hiện được sections
    """
    # 1. Trích xuất với phát hiện cột
    if method == 'smart':
        raw_text = extract_pdf_smart_layout(pdf_path)
    elif method == 'pdfplumber':
        raw_text = extract_pdf_pdfplumber_columns(pdf_path, column_count=2)
    else:  # auto
        raw_text = extract_pdf_with_column_detection(pdf_path, column_count='auto')
    
    # 2. Làm sạch text
    cleaned_text = clean_extracted_text(raw_text)
    
    # 3. Sắp xếp lại thành single column logic
    structured_text = restructure_to_single_column(cleaned_text, keep_original_if_no_sections=keep_original)
    
    return {
        'raw': raw_text,
        'cleaned': cleaned_text,
        'structured': structured_text
    }


def clean_extracted_text(text):
    """Làm sạch text đã trích xuất"""
    # Xóa dấu phân cách cột
    text = re.sub(r'=== CỘT (TRÁI|PHẢI|GIỮA) ===', '', text)
    text = re.sub(r'=== (HEADER|NỘI DUNG CHÍNH|THÔNG TIN BÊN) ===', '', text)
    
    # Xóa dấu phân trang
    text = re.sub(r'--- Trang \d+ ---', '', text)
    
    # Xóa khoảng trắng thừa
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def restructure_to_single_column(text, keep_original_if_no_sections=True):
    """
    Sắp xếp lại text thành single column theo logic CV
    Thứ tự: Tên → Contact → Mục tiêu → Học vấn → Kinh nghiệm → Kỹ năng → Dự án
    
    Args:
        text: Text đã trích xuất
        keep_original_if_no_sections: Nếu True, giữ nguyên text gốc khi không tìm thấy sections
    """
    sections = {
        'header': [],
        'contact': [],
        'objective': [],
        'education': [],
        'experience': [],
        'skills': [],
        'projects': [],
        'certifications': [],
        'languages': [],
        'awards': [],
        'other': []
    }
    
    # Keywords mở rộng hơn - hỗ trợ nhiều format CV
    section_keywords = {
        'objective': [
            'mục tiêu', 'objective', 'career objective', 'mong muốn',
            'career goal', 'professional summary', 'tóm tắt'
        ],
        'education': [
            'học vấn', 'education', 'đại học', 'university', 'college',
            'trường', 'training', 'academic', 'qualification'
        ],
        'experience': [
            'kinh nghiệm', 'experience', 'work experience', 'employment',
            'làm việc', 'work history', 'professional experience',
            'công ty', 'company', 'thực tập', 'internship'
        ],
        'skills': [
            'kỹ năng', 'skills', 'technical skills', 'competencies',
            'expertise', 'abilities', 'chuyên môn'
        ],
        'projects': [
            'dự án', 'projects', 'portfolio', 'work samples'
        ],
        'certifications': [
            'chứng chỉ', 'certifications', 'certificates', 'licenses',
            'bằng cấp', 'credentials'
        ],
        'languages': [
            'ngoại ngữ', 'languages', 'tiếng anh', 'foreign language',
            'language skills'
        ],
        'awards': [
            'giải thưởng', 'awards', 'honors', 'achievements',
            'recognition', 'thành tích'
        ]
    }
    
    lines = text.split('\n')
    current_section = 'other'
    section_found_count = 0
    
    # Track xem có tìm thấy ít nhất 1 keyword section không
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        line_lower = line_stripped.lower()
        
        # Kiểm tra xem dòng này có phải là section header không
        is_section_header = False
        
        # Ưu tiên: dòng ngắn, in hoa, hoặc có format đặc biệt
        is_potential_header = (
            len(line_stripped) < 60 and 
            (line_stripped.isupper() or 
             line_stripped.count(' ') < 5 or
             line_stripped.startswith('#'))
        )
        
        # Phát hiện section dựa trên keywords
        for section_name, keywords in section_keywords.items():
            if any(kw in line_lower for kw in keywords):
                # Nếu là dòng ngắn/header format → đây là section header
                if is_potential_header:
                    current_section = section_name
                    section_found_count += 1
                    is_section_header = True
                    break
                # Nếu không phải header nhưng có keyword → vẫn chuyển section
                elif len([kw for kw in keywords if kw in line_lower]) >= 1:
                    current_section = section_name
                    section_found_count += 1
        
        # Không thêm dòng section header vào kết quả
        if is_section_header:
            continue
        
        # Phát hiện contact info (email, phone, address)
        has_phone = bool(re.search(r'0\d{9,10}', line_stripped))
        has_email = '@' in line_stripped and '.' in line_stripped
        has_address_keywords = any(kw in line_lower for kw in [
            'hà nội', 'hồ chí minh', 'đà nẵng', 'quận', 'phường', 
            'district', 'ward', 'street', 'address'
        ])
        
        if has_phone or has_email or has_address_keywords:
            # Nếu chưa có section nào, đưa vào contact
            if current_section == 'other' or current_section == 'header':
                current_section = 'contact'
        
        # Thêm vào section tương ứng
        sections[current_section].append(line_stripped)
    
    # Nếu không tìm thấy sections và flag = True → giữ nguyên text gốc
    if section_found_count < 2 and keep_original_if_no_sections:
        return text
    
    # Ghép lại theo thứ tự chuẩn
    result = []
    
    # Header (thường là tên, title)
    if sections['header']:
        result.extend(sections['header'])
        result.append('')
    
    # Contact info
    if sections['contact']:
        result.append('## THÔNG TIN LIÊN HỆ')
        result.extend(sections['contact'])
        result.append('')
    
    # Main sections theo thứ tự logic
    section_order = [
        ('objective', '## MỤC TIÊU NGHỀ NGHIỆP'),
        ('education', '## HỌC VẤN'),
        ('experience', '## KINH NGHIỆM LÀM VIỆC'),
        ('skills', '## KỸ NĂNG'),
        ('projects', '## DỰ ÁN'),
        ('certifications', '## CHỨNG CHỈ'),
        ('languages', '## NGOẠI NGỮ'),
        ('awards', '## GIẢI THƯỞNG')
    ]
    
    for section_name, section_title in section_order:
        if sections[section_name]:
            result.append(section_title)
            result.extend(sections[section_name])
            result.append('')
    
    # Other - những phần không phân loại được
    if sections['other']:
        result.append('## THÔNG TIN KHÁC')
        result.extend(sections['other'])
    
    return '\n'.join(result)


# ============================================================================
# SỬ DỤNG - VÍ DỤ CHO FILE CV CỦA BẠN
# ============================================================================

pdf_file = "/kaggle/input/cv-huy/nguyen-duc-huy_1758517966_Joboko_c3e1a50bcfd6fb7f_3487225.pdf"

print("=== PHƯƠNG PHÁP 1: AUTO DETECT ===")
result1 = parse_cv_with_column_detection(pdf_file, method='auto')
print(result1['structured'])

print("\n\n=== PHƯƠNG PHÁP 2: SMART LAYOUT ===")
result2 = parse_cv_with_column_detection(pdf_file, method='smart')
print(result2['structured'])


# Lưu kết quả tốt nhất
best_result = result2  # Thường smart layout cho kết quả tốt nhất


# So sánh kết quả
print("\n=== SO SÁNH PHƯƠNG PHÁP ===")
print(f"Auto detect: {len(result1['structured'])} ký tự")
print(f"Smart layout: {len(result2['structured'])} ký tự")

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
prompt = """
     lọc ra từ CV, hãy đọc và Trích xuất thông tin chuẩn chỉ trong văn bản:
        - "skills": danh sách kỹ năng(dạng list)
       - "experience_years": số năm kinh nghiệm (số nguyên)
        - "education": bằng cấp cao nhất (chuỗi)
        - "certificates": danh sách chứng chỉ có (dạng list)
        - "project":danh sách các dự án đã làm
    Sau đó tính matching score với JD theo từng các tiêu chí skills, experience_years, education, certificates mỗi cái gồm 3 mục:
        -"score": điểm của tiêu chí đó
        -"missing": các thứ còn thiếu thiếu
        -"match": cac thứ đáp ứng được (nếu vượt qua yêu cầu JD thì cộng thêm chút điểm)
    Cuối cùng đưa ra điểm matching tổng thế, lưu ý tùy theo JD yêu cầu đặc biệt đối với tiêu chí nào thì cho tiêu chí đó hiệu số cao lên trong tính điểm matching cuối
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
