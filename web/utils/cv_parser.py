import re
from typing import List, Dict, Any, Optional

# Basic dictionaries for detection
TECH_KEYWORDS = [
    # Languages
    'python','java','javascript','typescript','c','c++','c#','go','golang','rust','ruby','php','kotlin','swift','scala','matlab','r','sql','no sql','nosql','powershell','bash','shell',
    # Frameworks/libs
    'django','flask','fastapi','spring','spring boot','react','react.js','reactjs','vue','vue.js','angular','node','node.js','express','.net','.net core','asp.net','laravel','rails','tensorflow','pytorch','keras','sklearn','scikit-learn','pandas','numpy','opencv','langchain','hugging face','transformers','spark','hadoop','hive','airflow','kafka',
    # Cloud/DevOps
    'aws','azure','gcp','google cloud','docker','kubernetes','terraform','ansible','jenkins','gitlab ci','github actions','ci/cd','linux','windows','macos','git',
    # Databases/BI
    'mysql','postgres','postgresql','mssql','sql server','sqlite','oracle','mongodb','redis','elasticsearch','bigquery','snowflake','redshift','power bi','tableau','looker',
    # Others
    'rest','graphql','grpc','microservices','message queue','rabbitmq','celery','oop','design patterns','unit test','pytest','junit','selenium','cypress','jmeter',
]

SOFT_SKILLS = [
    'communication','teamwork','collaboration','leadership','problem solving','critical thinking','time management','adaptability','creativity','presentation','negotiation','mentoring','coaching','self-motivated','proactive','agile','scrum','kanban','stakeholder management','analytical','organization','planning'
]

CERT_PATTERNS = [
    r'aws\s*(certified)?\s*(solutions|developer|sysops|data).*',
    r'azure\s*(administrator|developer|solutions|data).*',
    r'google\s*professional\s*(cloud|data|ml).*',
    r'oca|ocp|oracle\s*certified',
    r'ccna|ccnp|cissp|security\+|network\+',
    r'pmp|prince2',
    r'scrum\s*(master|product\s*owner)|psm\d',
    r'ielts|toefl|toeic',
]

DEGREE_WORDS = [
    'bachelor','master','phd','doctor','engineer','engineer degree','associate','diploma'
]

DEGREE_REGEX = re.compile(
    r'(?P<degree>bachelor|master|phd|doctor(ate)?|engineer|associate|diploma)\s*(of)?\s*(?P<major>[a-zA-Z&\-/\s]{0,60})',
    re.IGNORECASE
)

EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}')
PHONE_REGEX = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
GPA_REGEX = re.compile(r'GPA\s*[:\-]?\s*(\d\.?\d{0,2})\s*(/\s*10|/\s*4|/\s*100)?', re.IGNORECASE)
YEAR_REGEX = re.compile(r'(19|20)\d{2}')
DATE_RANGE_REGEX = re.compile(
    r'(?P<from>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{4}|(19|20)\d{2}|present)\s*(?:\-|to|–|—|>)\s*(?P<to>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}|present|(19|20)\d{2})',
    re.IGNORECASE
)

SECTION_MARKERS = {
    'education': re.compile(r'^\s*(education|học\s*vấn|academic)\s*$', re.IGNORECASE),
    'experience': re.compile(r'^\s*(experience|work\s*experience|kinh\s*nghiệm)\s*$', re.IGNORECASE),
    'skills': re.compile(r'^\s*(skills?|kỹ\s*năng)\s*$', re.IGNORECASE),
    'certifications': re.compile(r'^\s*(certifications?|chứng\s*chỉ)\s*$', re.IGNORECASE),
}


def _normalize_lines(text: str) -> List[str]:
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    lines = [re.sub(r'\s+', ' ', ln.strip()) for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _find_name(lines: List[str]) -> Optional[str]:
    # heuristic: first non-empty line with >=2 words, not an email or phone
    for ln in lines[:8]:
        if EMAIL_REGEX.search(ln) or PHONE_REGEX.search(ln):
            continue
        if len(ln.split()) >= 2 and len(ln) <= 60:
            # Avoid section headers
            if not any(m.search(ln) for m in SECTION_MARKERS.values()):
                return ln
    return None


def _extract_list_by_keywords(text: str, vocab: List[str]) -> List[str]:
    low = text.lower()
    found = []
    for kw in vocab:
        if kw in low:
            found.append(kw)
    # dedupe, preserve order
    seen = set()
    result = []
    for x in found:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _split_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {'education': [], 'experience': [], 'skills': [], 'certifications': [], 'other': []}
    current = 'other'
    for ln in lines:
        switched = False
        for key, rx in SECTION_MARKERS.items():
            if rx.search(ln):
                current = key
                switched = True
                break
        if switched:
            continue
        sections.setdefault(current, []).append(ln)
    return sections


def extract_cv_info(text: str) -> Dict[str, Any]:
    """
    Extract structured information from raw CV text.

    Returns dict with keys: personal_info, skills, education, experience, certifications
    """
    if not text:
        return {
            'personal_info': {'name': None, 'email': None, 'phone': None, 'address': None},
            'skills': {'technical': [], 'soft': []},
            'certifications': [],
            'education': [],
            'experience': []
        }

    lines = _normalize_lines(text)
    full_text = '\n'.join(lines)

    # Personal info
    emails = EMAIL_REGEX.findall(full_text)
    phones = PHONE_REGEX.findall(full_text)
    name = _find_name(lines)

    # rudimentary address: look for lines containing street/city keywords
    address = None
    for ln in lines[:20]:
        if any(tok in ln.lower() for tok in ['street', 'st.', 'district', 'ward', 'city', 'province', 'vietnam', 'viet nam']):
            address = ln
            break

    # Sections split
    sections = _split_sections(lines)

    # Skills
    tech_skills = set(_extract_list_by_keywords(full_text, TECH_KEYWORDS))
    soft_skills = set(_extract_list_by_keywords(full_text, SOFT_SKILLS))

    # Try to parse explicit skills lists in skills section
    for ln in sections.get('skills', []):
        for item in re.split(r'[•\u2022,;/\|]', ln):
            w = item.strip()
            lw = w.lower()
            if not w:
                continue
            if any(k in lw for k in TECH_KEYWORDS):
                tech_skills.add(lw)
            elif any(k in lw for k in SOFT_SKILLS):
                soft_skills.add(lw)

    # Certifications
    certs: List[str] = []
    for pat in CERT_PATTERNS:
        for m in re.finditer(pat, full_text, re.IGNORECASE):
            val = m.group(0)
            if val and val.lower() not in [c.lower() for c in certs]:
                certs.append(val)

    # Education
    education: List[Dict[str, Any]] = []
    edu_block = sections.get('education') or []
    if not edu_block:
        # Fallback: search entire text
        edu_block = lines
    for ln in edu_block:
        degm = DEGREE_REGEX.search(ln)
        if degm:
            degree = degm.group('degree')
            major = (degm.group('major') or '').strip(' -,:')
            years = YEAR_REGEX.findall(ln)
            year = None
            if years:
                year = max(int(y if len(y)==4 else '19'+y) if isinstance(y, str) else int(y) for y in [e if isinstance(e, str) else ''.join(e) for e in years])
            gpa = None
            gpam = GPA_REGEX.search(ln)
            if gpam:
                gpa = gpam.group(1)
            education.append({
                'institution': None,
                'degree': degree,
                'major': major if major else None,
                'graduation_year': year,
                'gpa': gpa,
                'description': ln
            })
        else:
            # Try to capture institution names heuristically
            if any(tok in ln.lower() for tok in ['university', 'academy', 'college', 'institute', 'hoc vien', 'đại học', 'dai hoc']):
                education.append({
                    'institution': ln,
                    'degree': None,
                    'major': None,
                    'graduation_year': None,
                    'gpa': None,
                    'description': ln
                })

    # Experience
    experience: List[Dict[str, Any]] = []
    exp_block = sections.get('experience') or []
    if not exp_block:
        exp_block = lines

    for i, ln in enumerate(exp_block):
        # Match date range first
        dr = DATE_RANGE_REGEX.search(ln)
        if dr:
            # Look around this line for title/company
            title = None
            company = None
            desc_lines: List[str] = []

            # Title/company heuristics
            if i > 0:
                prev = exp_block[i-1]
                if len(prev.split()) <= 12:
                    title = prev
            if i+1 < len(exp_block):
                nxt = exp_block[i+1]
                if len(nxt.split()) <= 12 and not DATE_RANGE_REGEX.search(nxt):
                    company = nxt

            # Collect description bullets until next date or blank/section
            j = i + 2
            while j < len(exp_block):
                l2 = exp_block[j]
                if DATE_RANGE_REGEX.search(l2) or SECTION_MARKERS['education'].search(l2) or SECTION_MARKERS['skills'].search(l2):
                    break
                if l2.strip():
                    desc_lines.append(l2)
                j += 1

            # Extract technologies from description
            tech_in_desc = list(_extract_list_by_keywords('\n'+"\n".join(desc_lines)+'\n', TECH_KEYWORDS))

            experience.append({
                'company': company,
                'title': title,
                'duration': dr.group(0),
                'from': dr.group('from'),
                'to': dr.group('to'),
                'technologies': tech_in_desc,
                'description': '\n'.join(desc_lines) if desc_lines else None
            })

    # Deduplicate education entries by description
    unique_edu = []
    seen_desc = set()
    for e in education:
        key = (e['degree'], e['major'], e['institution'], e['graduation_year'], e['gpa'])
        if key not in seen_desc:
            seen_desc.add(key)
            unique_edu.append(e)

    result: Dict[str, Any] = {
        'personal_info': {
            'name': name,
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'address': address,
        },
        'skills': {
            'technical': sorted(list(tech_skills)),
            'soft': sorted(list(soft_skills)),
        },
        'certifications': certs,
        'education': unique_edu[:10],
        'experience': experience[:20]
    }

    return result

if __name__ == '__main__':
    sample = """
NGUYEN DUC HUY
Email: huydcb3@gmail.com | Phone: 0962865038 | Ho Chi Minh City, Vietnam

Skills: Python, C++, Git, Linux, TensorFlow, PyTorch, LangChain, Hugging Face, Docker, Kubernetes, Communication, Teamwork

Experience
01/2022 - 03/2025
Dev Tool Python, HEXAGON Co., Ltd
- Build internal dev tools with Python, FastAPI, Git and CI/CD on GitHub Actions
- Applied LangChain and Hugging Face models for document processing

Education
Bachelor of Computer Science, HCM University of Technology, 2021 | GPA: 3.4/4
Certifications: AWS Certified Cloud Practitioner, IELTS 7.0
"""
    import json
    print(json.dumps(extract_cv_info(sample), ensure_ascii=False, indent=2))

