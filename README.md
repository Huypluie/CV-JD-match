# 🧠 AI Resume–JD Matching System 
**Link Web :**  
[https://drive.google.com/drive/folders/1gSw5u1YX_5p-TaxxFqOlBupnjzkHve_9?usp=sharing  ](https://drive.google.com/drive/folders/1VXtPhAp4FFV2sMW-LBrpM-x969KPniKf?usp=sharing)
## 📌 Giới thiệu

Dự án này tự động:
1. **Trích xuất văn bản từ CV PDF có layout phức tạp (2–3 cột)**.
2. **Chuẩn hóa, tái cấu trúc nội dung CV** thành dạng logic.
3. **Ẩn thông tin cá nhân** như tên, email, địa chỉ, số điện thoại, ngày sinh.
4. **Phân tích nội dung CV bằng LLM (Gemini / Qwen)** → trích xuất kỹ năng, học vấn, chứng chỉ, dự án, kinh nghiệm.
5. **So khớp với JD (Job Description)** để tính điểm matching tự động theo từng tiêu chí:
   - Kỹ năng (`skills`)
   - Kinh nghiệm (`experience_years`)
   - Học vấn (`education`)
   - Chứng chỉ (`certificates`)
6. **Xuất ra JSON chứa điểm chi tiết + điểm tổng hợp (overall_score).**

---
## ⚙️ Kiến trúc pipeline

```mermaid
graph TD
    A[📄 CV PDF] --> B[📘 PyMuPDF + PDFPlumber: Trích xuất văn bản]
    B --> C[🧩 LayoutLMv3: Giữ bố cục + embedding 768D]
    C --> D[🧹 Làm sạch & tái cấu trúc CV]
    D --> E[ Qwen2-7B:trích xuất và xóa thông tin cá nhân ]
    E --> F[🧠 Gemini Pro: Trích xuất structured JSON]
    F --> G[📊 Gemini Pro: Tính điểm matching với JD]
    G --> H[✅ Xuất JSON kết quả + Matching Score]


