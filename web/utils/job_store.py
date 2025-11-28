import os
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'jobs.json')

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize DB if missing
if not os.path.exists(DB_PATH):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump({'jobs': []}, f, ensure_ascii=False, indent=2)


def _load_db() -> Dict[str, Any]:
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_db(db: Dict[str, Any]):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def list_jobs() -> List[Dict[str, Any]]:
    return _load_db().get('jobs', [])


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    for job in list_jobs():
        if job.get('id') == job_id:
            return job
    return None


def create_job(name: str, jd_file_path: str, jd_text: str) -> Dict[str, Any]:
    db = _load_db()
    job_id = str(uuid.uuid4())
    job = {
        'id': job_id,
        'name': name,
        'jd_file': jd_file_path,
        'jd_text': jd_text,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'cvs': []
    }
    db.setdefault('jobs', []).append(job)
    _save_db(db)
    return job


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    db = _load_db()
    for job in db.get('jobs', []):
        if job.get('id') == job_id:
            job.update(updates or {})
            _save_db(db)
            return job
    return None


def add_cv(job_id: str, file_name: str, file_path: str, extracted_text: str) -> Dict[str, Any]:
    db = _load_db()
    for job in db.get('jobs', []):
        if job.get('id') == job_id:
            cv = {
                'id': str(uuid.uuid4()),
                'file_name': file_name,
                'file_path': file_path,
                'extracted_text': extracted_text,
                'parsed_data': None,
                'score_report': None,
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
            job.setdefault('cvs', []).append(cv)
            _save_db(db)
            return cv
    raise ValueError('Job not found')


def update_cv_parsed(job_id: str, cv_id: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    db = _load_db()
    for job in db.get('jobs', []):
        if job.get('id') == job_id:
            for cv in job.get('cvs', []):
                if cv.get('id') == cv_id:
                    cv['parsed_data'] = parsed
                    _save_db(db)
                    return cv
    raise ValueError('CV not found')


def update_cv_score(job_id: str, cv_id: str, score_report: Dict[str, Any]) -> Dict[str, Any]:
    db = _load_db()
    for job in db.get('jobs', []):
        if job.get('id') == job_id:
            for cv in job.get('cvs', []):
                if cv.get('id') == cv_id:
                    cv['score_report'] = score_report
                    _save_db(db)
                    return cv
    raise ValueError('CV not found')


def list_cvs(job_id: str) -> List[Dict[str, Any]]:
    job = get_job(job_id)
    return job.get('cvs', []) if job else []


def delete_job(job_id: str) -> bool:
    db = _load_db()
    jobs = db.get('jobs', [])
    before = len(jobs)
    db['jobs'] = [j for j in jobs if j.get('id') != job_id]
    _save_db(db)
    return len(db['jobs']) < before




def delete_cv(job_id: str, cv_id: str) -> bool:
    db = _load_db()
    for job in db.get('jobs', []):
        if job.get('id') == job_id:
            cvs = job.get('cvs', [])
            before = len(cvs)
            job['cvs'] = [c for c in cvs if c.get('id') != cv_id]
            _save_db(db)
            return len(job['cvs']) < before
    return False


def replace_cv(job_id: str, cv_id: str, file_name: str, file_path: str, extracted_text: str) -> Optional[Dict[str, Any]]:
    db = _load_db()
    for job in db.get('jobs', []):
        if job.get('id') == job_id:
            for cv in job.get('cvs', []):
                if cv.get('id') == cv_id:
                    cv.update({
                        'file_name': file_name,
                        'file_path': file_path,
                        'extracted_text': extracted_text,
                        'parsed_data': None,
                        'score_report': None,
                    })
                    _save_db(db)
                    return cv
    return None
