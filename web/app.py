import os
import json
import re
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import google.generativeai as genai
from docx import Document
from dotenv import load_dotenv
from utils.cv_parser import extract_cv_info
from utils.job_store import list_jobs, create_job, get_job, add_cv, list_cvs, update_cv_parsed, update_cv_score, delete_job, update_job, delete_cv, replace_cv

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Vui lòng thiết lập GEMINI_API_KEY trong file .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Change this in production

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")

def clean_json_response(response_text):
    """Clean and extract JSON from Gemini response"""
    try:
        # Remove markdown code blocks if present
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()

        # Try to parse JSON
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response from AI: {str(e)}")

def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting for plain-text replies."""
    try:
        s = str(text)
        # Remove fenced code blocks
        s = re.sub(r"```[\s\S]*?```", "", s)
        # Remove inline code backticks
        s = s.replace("`", "")
        # Bold/italics variants
        s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
        s = re.sub(r"__(.*?)__", r"\1", s)
        s = re.sub(r"\*(.*?)\*", r"\1", s)
        s = re.sub(r"_(.*?)_", r"\1", s)
        # Strip markdown headers at line start
        s = re.sub(r"^\s{0,3}#+\s*", "", s, flags=re.MULTILINE)
        # Collapse excessive blank lines
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()
    except Exception:
        return text

def parse_cv_with_gemini(text):
    """Parse CV text using Gemini to extract structured information (strict JSON only)."""
    try:
        prompt = (
            "You are an expert CV parser. Read the CV text and return ONLY one valid JSON object following EXACTLY the schema below. "
            "Do not include markdown, commentary or any extra text. Assume the CV may be Vietnamese or English. "
            "Normalize skills to lowercase; deduplicate. Prefer data that actually appears in the CV; never hallucinate.\n\n"
            "Schema:\n"
            "{\n"
            "  \"personal_info\": {\n"
            "    \"name\": \"string|null\",\n"
            "    \"date_of_birth\": \"string|null\",\n"
            "    \"gender\": \"string|null\",\n"
            "    \"address\": \"string|null\",\n"
            "    \"email\": \"string|null\",\n"
            "    \"phone\": \"string|null\"\n"
            "  },\n"
            "  \"education\": [{\n"
            "    \"degree\": \"string|null\",\n"
            "    \"institution\": \"string|null\",\n"
            "    \"year\": \"string|null\",\n"
            "    \"description\": \"string|null\"\n"
            "  }],\n"
            "  \"experience\": [{\n"
            "    \"title\": \"string|null\",\n"
            "    \"company\": \"string|null\",\n"
            "    \"duration\": \"string|null\",\n"
            "    \"description\": \"string|null\"\n"
            "  }],\n"
            "  \"skills\": {\n"
            "    \"technical\": [\"string\"],\n"
            "    \"soft\": [\"string\"]\n"
            "  },\n"
            "  \"certifications\": [\"string\"]\n"
            "}\n\n"
            "Rules:\n"
            "- email must be copied exactly as in CV; phone should include country/area codes if present.\n"
            "- skills arrays should be lowercase tokens (e.g., 'python', 'pytorch', 'hugging face', 'langchain').\n"
            "- Keep arrays reasonably short (<= 40 items each).\n"
            "- If a field is unknown, return null.\n\n"
            "CV Text:\n" + text[:10000]
        )
        response = model.generate_content(prompt)
        return clean_json_response(response.text)
    except Exception as e:
        return {"error": f"Failed to parse CV: {str(e)}"}

def parse_jd_with_gemini(text):
    """Parse Job Description text using Gemini to extract structured information"""
    try:
        prompt = """Extract the following information from the Job Description text below and return ONLY a valid JSON object with these exact fields:
{
    "job_title": "string",
    "company": "string or null",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "responsibilities": ["responsibility1", "responsibility2"],
    "requirements": ["requirement1", "requirement2"],
    "experience_level": "string",
    "education": "string or null"
}

Job Description Text:
""" + text[:10000]  # Limit text length

        response = model.generate_content(prompt)
        return clean_json_response(response.text)
    except Exception as e:
        return {"error": f"Failed to parse JD: {str(e)}"}

def calculate_matching_score(cv_data, jd_data):
    """Rule-based matching based on extracted skills (fallback)."""
    try:
        if not isinstance(cv_data, dict):
            cv_data = {}
        if not isinstance(jd_data, dict):
            jd_data = {}

        # Extract skills from CV
        cv_skills = set()
        if isinstance(cv_data.get('skills'), list):
            cv_skills = set(str(skill).lower().strip() for skill in cv_data['skills'] if str(skill).strip())

        # Extract skills from JD
        jd_required_skills = set()
        jd_preferred_skills = set()

        if isinstance(jd_data.get('required_skills'), list):
            jd_required_skills = set(str(skill).lower().strip() for skill in jd_data['required_skills'] if str(skill).strip())
        if isinstance(jd_data.get('preferred_skills'), list):
            jd_preferred_skills = set(str(skill).lower().strip() for skill in jd_data['preferred_skills'] if str(skill).strip())

        # Calculate matches
        required_matches = cv_skills.intersection(jd_required_skills)
        preferred_matches = cv_skills.intersection(jd_preferred_skills)

        # Calculate score (weighted: required skills count more)
        required_score = len(required_matches) * 3  # Higher weight
        preferred_score = len(preferred_matches) * 1  # Lower weight
        total_possible = (len(jd_required_skills) * 3) + len(jd_preferred_skills)

        if total_possible > 0:
            score = min(int(((required_score + preferred_score) / total_possible) * 100), 100)
        else:
            score = 0

        return {
            'score': score,
            'matched_skills': list(required_matches.union(preferred_matches)),
            'missing_required_skills': list(jd_required_skills - cv_skills),
            'missing_preferred_skills': list(jd_preferred_skills - cv_skills),
            'job_title': jd_data.get('job_title', 'N/A'),
            'company': jd_data.get('company', 'N/A'),
            'experience_level': jd_data.get('experience_level', 'Not specified'),
            'education_required': jd_data.get('education', 'Not specified')
        }
    except Exception as e:
        return {"error": str(e), "score": 0}


def gemini_match(cv_text: str, jd_text: str):
    """Use Gemini to compute holistic match score and rationale."""
    try:
        prompt = f"""
You are an ATS/job-matching engine. Compare the candidate CV and the Job Description and respond ONLY with a valid JSON object with these exact fields:
{{
  "score": 0-100 integer,
  "matched_skills": ["skill"...],
  "missing_skills": ["skill"...],
  "strengths": ["point"...],
  "gaps": ["point"...]
}}

Rules:
- Score must be an integer 0..100.
- matched_skills/missing_skills should be normalized (lowercase, no duplicates).
- Base primarily on skills/responsibilities/experience relevance.

CV:
{cv_text[:8000]}

JOB DESCRIPTION:
{jd_text[:8000]}
"""
        response = model.generate_content(prompt)
        res = clean_json_response(response.text)
        # Ensure types
        res['score'] = int(max(0, min(100, int(res.get('score', 0)))))
        res['matched_skills'] = list({str(s).lower() for s in res.get('matched_skills', []) if str(s).strip()})
        res['missing_skills'] = list({str(s).lower() for s in res.get('missing_skills', []) if str(s).strip()})
        res['strengths'] = [str(s) for s in res.get('strengths', [])][:10]
        res['gaps'] = [str(s) for s in res.get('gaps', [])][:10]
        return res
    except Exception as e:
        return {"error": f"Gemini match failed: {str(e)}"}



def gemini_detailed_match(cv_text: str, jd_text: str):
    """Use Gemini with a strict JSON-only prompt to produce a weighted ATS-style analysis."""
    try:
        prompt = (
            "You are an expert HR analyst and an AI-powered Applicant Tracking System (ATS). "
            "Your task is to meticulously analyze a candidate's CV against a provided Job Description (JD). "
            "First, extract key information from both documents. Then, perform a detailed matching analysis for four key criteria: "
            "Skills, Years of Experience, Education, and Certifications. Finally, calculate a weighted overall matching score.\n\n"
            "CV Text:\n" + cv_text[:8000] + "\n\n"
            "Job Description Text:\n" + jd_text[:8000] + "\n\n"
            "IMPORTANT: All strings in the JSON must be written in Vietnamese, regardless of the source language of the CV/JD. "
            "Translate summaries and rationales to Vietnamese. Skill tokens/names may keep their original language if they are proper nouns/technical terms. "
            "Based on your analysis, you must generate ONLY a single, valid JSON object with the exact structure and data types specified below. "
            "DO NOT include any explanations, markdown formatting, or any text outside of the JSON object.\n\n"
            "Required JSON Output Structure:\n"
            "{\n"
            "  \"overall_score\": {\n"
            "    \"score\": \"integer (0-100)\",\n"
            "    \"summary\": \"string (Tóm tắt 1 câu về mức độ phù hợp của ứng viên)\"\n"
            "  },\n"
            "  \"criteria_analysis\": {\n"
            "    \"skills\": {\n"
            "      \"score\": \"integer (0-100)\",\n"
            "      \"jd_requirements\": [\"string (Danh sách kỹ năng JD yêu cầu/ưu tiên)\"],\n"
            "      \"cv_matches\": [\"string (Kỹ năng trong CV trùng với JD)\"],\n"
            "      \"missing\": [\"string (Kỹ năng quan trọng trong JD còn thiếu)\"],\n"
            "      \"rationale\": \"string (Giải thích ngắn gọn cách chấm mục kỹ năng)\"\n"
            "    },\n"
            "    \"experience_years\": {\n"
            "      \"score\": \"integer (0-100)\",\n"
            "      \"jd_requirements\": \"string (VD: '5+ năm', 'Tối thiểu 3 năm kinh nghiệm vị trí tương tự')\",\n"
            "      \"cv_value\": \"integer (Số năm kinh nghiệm liên quan trích xuất từ CV)\",\n"
            "      \"rationale\": \"string (Giải thích ngắn, VD: 'Đủ 5 năm' hoặc 'Chỉ 2 năm, thiếu so với yêu cầu 5')\"\n"
            "    },\n"
            "    \"education\": {\n"
            "      \"score\": \"integer (0-100)\",\n"
            "      \"jd_requirements\": \"string (Bằng cấp/trình độ tối thiểu trong JD, VD: 'Cử nhân CNTT')\",\n"
            "      \"cv_matches\": \"string (Bằng cấp phù hợp cao nhất trong CV)\",\n"
            "      \"rationale\": \"string (Giải thích ngắn về mức độ khớp/thiếu)\"\n"
            "    },\n"
            "    \"certificates\": {\n"
            "      \"score\": \"integer (0-100)\",\n"
            "      \"jd_requirements\": [\"string (Danh sách chứng chỉ JD yêu cầu/ưu tiên)\"],\n"
            "      \"cv_matches\": [\"string (Chứng chỉ trong CV phù hợp JD)\"],\n"
            "      \"missing\": [\"string (Chứng chỉ quan trọng trong JD còn thiếu)\"],\n"
            "      \"rationale\": \"string (Giải thích ngắn về mức độ khớp/thiếu)\"\n"
            "    }\n"
            "  }\n"
            "}\n\n"
            "Instructions for Populating the JSON:\n"
            "1) overall_score.score: Calculate weighted average with weights: skills 50%, experience_years 30%, education 10%, certificates 10%. Return an integer 0..100.\n"
            "2) Follow the exact schema and types. Return ONLY the JSON object, no extra text.\n"
        )
        response = model.generate_content(prompt)
        return clean_json_response(response.text)
    except Exception as e:
        return {"error": f"Gemini detailed match failed: {str(e)}"}

@app.route('/')
def jobs_home():
    return render_template('code.html')

@app.route('/jobs-page')
def jobs_page():
    return render_template('jobs.html', jobs=list_jobs())

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/jobs/<job_id>/view')
def jobs_view(job_id):
    job = get_job(job_id)
    if not job:
        return "Không tìm thấy công việc", 404
    return render_template('job_detail.html', job=job)

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Chưa tải lên tệp nào'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn tệp nào'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Định dạng tệp không được phép. Vui lòng tải lên tệp PDF, DOC hoặc DOCX.'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                extracted_text = extract_text_from_pdf(filepath)
            elif filename.lower().endswith(('.doc', '.docx')):
                extracted_text = extract_text_from_docx(filepath)
            else:
                return jsonify({'error': 'Định dạng tệp không được hỗ trợ'}), 400

            if not extracted_text.strip():
                return jsonify({'error': 'Không thể trích xuất văn bản từ tệp'}), 400

            # Parse CV text into structured data and store
            cv_struct = extract_cv_info(extracted_text)
            session['cv_text'] = extracted_text[:12000]
            session['cv_data'] = cv_struct

            return jsonify({
                'status': 'success',
                'cv_data': cv_struct,
                'extracted_text': extracted_text[:1000] + '...' if len(extracted_text) > 1000 else extracted_text
            })

        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        return jsonify({'error': f'Lỗi máy chủ: {str(e)}'}), 500

@app.route('/upload-jd', methods=['POST'])
def upload_jd():
    try:
        jd_text = ""

        # Check if it's file upload or text input
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']

            if not allowed_file(file.filename):
                return jsonify({'error': 'Định dạng tệp không được phép. Vui lòng tải lên tệp PDF, DOC hoặc DOCX.'}), 400

            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Extract text based on file type
                if filename.lower().endswith('.pdf'):
                    jd_text = extract_text_from_pdf(filepath)
                elif filename.lower().endswith(('.doc', '.docx')):
                    jd_text = extract_text_from_docx(filepath)
                else:
                    return jsonify({'error': 'Định dạng tệp không được hỗ trợ'}), 400
            finally:
                # Clean up the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

        elif 'jd_text' in request.form:
            jd_text = request.form.get('jd_text', '').strip()
        else:
            return jsonify({'error': 'Không có mô tả công việc nào được cung cấp'}), 400

        if not jd_text.strip():
            return jsonify({'error': 'Không thể trích xuất hoặc cung cấp văn bản'}), 400

        # Do NOT parse JD at upload time. Just store text (truncate for session size)
        session['jd_text'] = jd_text[:12000]
        # Keep minimal jd_data for backward compatibility
        session['jd_data'] = {}

        return jsonify({
            'status': 'success',
            'jd_data': {},
            'jd_text': jd_text[:1000] + '...' if len(jd_text) > 1000 else jd_text
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi máy chủ: {str(e)}'}), 500

@app.route('/calculate-matching', methods=['POST'])
def calculate_matching():
    try:
        cv_data = session.get('cv_data')
        jd_data = session.get('jd_data')
        cv_text = session.get('cv_text')
        jd_text = session.get('jd_text')

        if not cv_data or not cv_text:
            return jsonify({'error': 'Vui lòng tải lên CV của bạn trước'}), 400
        if not jd_data or not jd_text:
            return jsonify({'error': 'Vui lòng tải lên mô tả công việc trước'}), 400

        # Try Gemini holistic matching first
        ai_result = gemini_match(cv_text, jd_text)
        if isinstance(ai_result, dict) and 'error' not in ai_result and 'score' in ai_result:
            score = int(ai_result.get('score', 0))
            matched_skills = ai_result.get('matched_skills', [])
            missing_skills = ai_result.get('missing_skills', [])
        else:
            # Fallback to rule-based matching
            rb = calculate_matching_score(cv_data, jd_data)
            score = int(rb.get('score', 0))
            matched_skills = rb.get('matched_skills', [])
            missing_skills = rb.get('missing_required_skills', []) + rb.get('missing_preferred_skills', [])

        return jsonify({
            'status': 'success',
            'score': score,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'job_title': jd_data.get('job_title', 'N/A'),
            'company': jd_data.get('company', 'N/A')
        })
    except Exception as e:
        return jsonify({'error': f'Lỗi máy chủ: {str(e)}'}), 500

# ================= Job-centric JSON API =================
@app.route('/jobs', methods=['GET'])
def api_list_jobs():
    try:
        return jsonify({'jobs': list_jobs()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs', methods=['POST'])
def api_create_job():
    try:
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Tên công việc là bắt buộc'}), 400
        if 'jd' not in request.files or request.files['jd'].filename == '':
            return jsonify({'error': 'Cần có tệp JD'}), 400
        file = request.files['jd']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Định dạng tệp không được phép. Vui lòng tải lên tệp PDF, DOC hoặc DOCX.'}), 400
        # Save temporary to extract
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp_' + filename)
        file.save(temp_path)
        try:
            if filename.lower().endswith('.pdf'):
                jd_text = extract_text_from_pdf(temp_path)
            elif filename.lower().endswith(('.doc', '.docx')):
                jd_text = extract_text_from_docx(temp_path)
            else:
                return jsonify({'error': 'Định dạng tệp không được hỗ trợ'}), 400
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        # Persist job and copy JD to uploads/jobs/<job_id>/
        job = create_job(name, '', jd_text)
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', job['id'])
        os.makedirs(job_dir, exist_ok=True)
        jd_path = os.path.join(job_dir, filename)
        # Save original JD upload
        request.files['jd'].save(jd_path)
        # Update job with JD file path in JSON store
        job = update_job(job['id'], {'jd_file': jd_path}) or job

        return jsonify({'job': job})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>', methods=['GET'])
def api_get_job(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Không tìm thấy công việc'}), 404
    return jsonify({'job': job})


@app.route('/jobs/<job_id>', methods=['DELETE'])
def api_delete_job(job_id):
    try:
        ok = delete_job(job_id)
        if not ok:
            return jsonify({'error': 'Không tìm thấy công việc'}), 404
        # remove files
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', job_id)
        if os.path.isdir(job_dir):
            for root, _, files in os.walk(job_dir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
            try:
                os.removedirs(job_dir)
            except Exception:
                pass
        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>', methods=['PATCH'])
def api_update_job(job_id):
    try:
        name = request.form.get('name', '').strip()
        upd = {}
        if name:
            upd['name'] = name
        job = update_job(job_id, upd)
        if not job:
            return jsonify({'error': 'Không tìm thấy công việc'}), 404
        return jsonify({'job': job})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>/jd', methods=['PATCH'])
def api_replace_jd(job_id):
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Không tìm thấy công việc'}), 404
        if 'jd' not in request.files or request.files['jd'].filename == '':
            return jsonify({'error': 'Cần có tệp JD'}), 400
        f = request.files['jd']
        if not allowed_file(f.filename):
            return jsonify({'error': 'Định dạng tệp không được phép'}), 400
        filename = secure_filename(f.filename)
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', job_id)
        os.makedirs(job_dir, exist_ok=True)
        save_path = os.path.join(job_dir, filename)
        f.save(save_path)
        # Extract text
        if filename.lower().endswith('.pdf'):
            jd_text = extract_text_from_pdf(save_path)
        elif filename.lower().endswith(('.doc', '.docx')):
            jd_text = extract_text_from_docx(save_path)
        else:
            return jsonify({'error': 'Định dạng tệp không được hỗ trợ'}), 400
        # Remove old JD file if exists and different path
        old_path = job.get('jd_file')
        if old_path and os.path.exists(old_path) and os.path.abspath(old_path) != os.path.abspath(save_path):
            try:
                os.remove(old_path)
            except Exception:
                pass
        updated = update_job(job_id, {'jd_file': save_path, 'jd_text': jd_text})
        return jsonify({'job': updated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>/cvs/<cv_id>', methods=['DELETE'])
def api_delete_cv(job_id, cv_id):
    try:
        # remove file on disk
        job, cv = _find_cv(job_id, cv_id)
        if not job:
            return jsonify({'error': 'Không tìm thấy công việc'}), 404
        if not cv:
            return jsonify({'error': 'Không tìm thấy CV'}), 404
        fp = cv.get('file_path')
        ok = delete_cv(job_id, cv_id)
        if ok and fp and os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>/cvs/<cv_id>', methods=['PUT'])
def api_replace_cv(job_id, cv_id):
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Không tìm thấy công việc'}), 404
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'Cần có tệp CV'}), 400
        f = request.files['file']
        if not allowed_file(f.filename):
            return jsonify({'error': 'Định dạng tệp không được phép'}), 400
        filename = secure_filename(f.filename)
        cv_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', job_id, 'cvs')
        os.makedirs(cv_dir, exist_ok=True)
        save_path = os.path.join(cv_dir, filename)
        f.save(save_path)
        # Extract text
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(save_path)
        elif filename.lower().endswith(('.doc', '.docx')):
            text = extract_text_from_docx(save_path)
        else:
            return jsonify({'error': 'Định dạng tệp không được hỗ trợ'}), 400
        # remove old file if exist
        _, cv = _find_cv(job_id, cv_id)
        old_path = cv.get('file_path') if cv else None
        updated = replace_cv(job_id, cv_id, filename, save_path, text)
        if not updated:
            return jsonify({'error': 'Không tìm thấy CV'}), 404
        if old_path and os.path.exists(old_path) and os.path.abspath(old_path) != os.path.abspath(save_path):
            try:
                os.remove(old_path)
            except Exception:
                pass
        return jsonify({'cv': updated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>/cvs', methods=['POST'])
def api_upload_cvs(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Không tìm thấy công việc'}), 404
    files = request.files.getlist('files') or ([request.files['file']] if 'file' in request.files else [])
    if not files:
        return jsonify({'error': 'Chưa tải lên tệp CV nào'}), 400
    created = []
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', job_id, 'cvs')
    os.makedirs(job_dir, exist_ok=True)
    for f in files:
        if not f.filename:
            continue
        if not allowed_file(f.filename):
            continue
        filename = secure_filename(f.filename)
        save_path = os.path.join(job_dir, filename)
        f.save(save_path)
        # Extract text
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(save_path)
        elif filename.lower().endswith(('.doc', '.docx')):
            text = extract_text_from_docx(save_path)
        else:
            continue
        cvrec = add_cv(job_id, filename, save_path, text)
        created.append(cvrec)
    return jsonify({'created': created})


@app.route('/jobs/<job_id>/cvs', methods=['GET'])
def api_list_cvs(job_id):
    return jsonify({'cvs': list_cvs(job_id)})


def _find_cv(job_id: str, cv_id: str):
    job = get_job(job_id)
    if not job:
        return None, None
    for cv in job.get('cvs', []):
        if cv.get('id') == cv_id:
            return job, cv
    return job, None


@app.route('/jobs/<job_id>/cvs/<cv_id>', methods=['GET'])
def api_get_cv(job_id, cv_id):
    job, cv = _find_cv(job_id, cv_id)
    if not job:
        return jsonify({'error': 'Không tìm thấy công việc'}), 404
    if not cv:
        return jsonify({'error': 'Không tìm thấy CV'}), 404
    return jsonify({'cv': cv})


@app.route('/jobs/<job_id>/cvs/<cv_id>/parse', methods=['POST'])
def api_parse_cv(job_id, cv_id):
    job, cv = _find_cv(job_id, cv_id)
    if not job:
        return jsonify({'error': 'Không tìm thấy công việc'}), 404
    if not cv:
        return jsonify({'error': 'Không tìm thấy CV'}), 404
    text = cv.get('extracted_text', '') or ''
    # Try LLM-based parsing first
    llm = parse_cv_with_gemini(text)
    if isinstance(llm, dict) and 'error' not in llm and llm.get('personal_info'):
        parsed = llm
    else:
        parsed = extract_cv_info(text)
    updated = update_cv_parsed(job_id, cv_id, parsed)
    return jsonify({'cv': updated})


@app.route('/jobs/<job_id>/cvs/<cv_id>/score', methods=['POST'])
def api_score_cv(job_id, cv_id):
    job, cv = _find_cv(job_id, cv_id)
    if not job:
        return jsonify({'error': 'Không tìm thấy công việc'}), 404
    if not cv:
        return jsonify({'error': 'Không tìm thấy CV'}), 404
    jd_text = job.get('jd_text', '')
    cv_text = cv.get('extracted_text', '')
    if not jd_text or not cv_text:
        return jsonify({'error': 'Thiếu văn bản JD hoặc CV'}), 400
    report = gemini_detailed_match(cv_text, jd_text)
    if isinstance(report, dict) and 'overall_score' in report:
        updated = update_cv_score(job_id, cv_id, report)
        return jsonify({'cv': updated})
    return jsonify({'error': report.get('error', 'AI thất bại')}), 500


@app.route('/ai-match-detailed', methods=['POST'])
def ai_match_detailed():
    try:
        cv_text = session.get('cv_text')
        jd_text = session.get('jd_text')
        if not cv_text:
            return jsonify({'error': 'Vui lòng tải lên CV của bạn trước'}), 400
        if not jd_text:
            return jsonify({'error': 'Vui lòng tải lên mô tả công việc trước'}), 400
        result = gemini_detailed_match(cv_text, jd_text)
        if isinstance(result, dict) and 'overall_score' in result:
            return jsonify(result)
        return jsonify({'error': 'AI không trả về đúng định dạng JSON mong đợi'}), 500
    except Exception as e:
        return jsonify({'error': f'Lỗi máy chủ: {str(e)}'}), 500


# ================= Chatbot API =================
@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get('message') or '').strip()
        if not user_message:
            return jsonify({'error': 'cần có tin nhắn'}), 400

        # Initialize history
        history = session.get('chat_history') or []

        # System/context priming (Vietnamese assistant for this app domain)
        system_prompt = (
            "Bạn là trợ lý AI tiếng Việt, hỗ trợ người dùng tuyển dụng: tải CV/JD, phân tích, so khớp kỹ năng, và hướng dẫn sử dụng ứng dụng. "
            "Trả lời ngắn gọn, súc tích, có gợi ý bước tiếp theo khi phù hợp. "
            "Luôn trả lời thuần văn bản (plain text), KHÔNG dùng markdown (ví dụ: **, _, #, `, danh sách)."
        )

        # Build conversation transcript
        transcript_parts = [f"System: {system_prompt}"]
        for turn in history:
            transcript_parts.append(f"User: {turn.get('user','')}")
            transcript_parts.append(f"Assistant: {turn.get('assistant','')}")
        transcript_parts.append(f"User: {user_message}")
        prompt = "\n\n".join(transcript_parts)

        # Build app data context snapshot
        cv_present = bool(session.get('cv_text'))
        jd_present = bool(session.get('jd_text'))
        # summarize cv data if available
        cv_data = session.get('cv_data') or {}
        cv_name = ((cv_data.get('personal_info') or {}).get('name')) if isinstance(cv_data, dict) else None

        # jobs overview
        jobs = list_jobs()
        jobs_count = len(jobs)
        last_jobs = [j.get('name') for j in jobs[-5:]] if isinstance(jobs, list) else []

        # optional page context: job_id
        ctx = data.get('context') or {}
        job_ctx = {}
        if isinstance(ctx, dict) and ctx.get('job_id'):
            j = get_job(ctx.get('job_id'))
            if j:
                job_ctx = {
                    'id': j.get('id'),
                    'name': j.get('name'),
                    'cv_count': len(j.get('cvs', [])),
                    'has_jd_file': bool(j.get('jd_file')),
                    'jd_excerpt': (j.get('jd_text') or '')[:600]
                }

        context_lines = [
            "[DỮ LIỆU ỨNG DỤNG]",
            f"Số job hiện có: {jobs_count}",
            f"Danh sách job gần đây: {', '.join([n for n in last_jobs if n])}" if last_jobs else "",
            f"Trong phiên: CV đã tải: {cv_present}, JD đã tải: {jd_present}",
            f"Tên ứng viên trong CV: {cv_name}" if cv_name else "",
        ]
        if job_ctx:
            context_lines.extend([
                "[JOB ĐANG XEM]",
                f"Job: {job_ctx.get('name')} (CVs: {job_ctx.get('cv_count')}, JD file: {job_ctx.get('has_jd_file')})",
                "JD (trích):\n" + (job_ctx.get('jd_excerpt') or '')
            ])

        context_note = "\n".join([l for l in context_lines if l]) + "\n\nHãy dùng dữ liệu trên để trả lời chính xác, ngắn gọn."

        response = model.generate_content(prompt + "\n\n" + context_note)
        assistant_reply = strip_markdown((response.text or '').strip())

        # Update history (truncate to avoid oversized session)
        history.append({'user': user_message, 'assistant': assistant_reply})
        if len(history) > 20:
            history = history[-20:]
        session['chat_history'] = history

        return jsonify({'reply': assistant_reply, 'history_len': len(history)})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/chat/reset', methods=['POST'])
def api_chat_reset():
    session['chat_history'] = []
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
