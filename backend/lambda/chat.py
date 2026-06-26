import json
import boto3
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from boto3.dynamodb.conditions import Attr
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
 
# Configuration
REGION = 'us-east-1'
OPENSEARCH_ENDPOINT = os.environ['OPENSEARCH_ENDPOINT']
INDEX_NAME = 'umd-knowledge'
COURSES_TABLE = 'umd-chatbot-courses'
PLANETTERP_BASE = 'https://api.planetterp.com/v1'
TOP_K = 3
MAX_PROFS_TO_ENRICH = 8  # Cap to keep latency sane
 
# AWS clients
bedrock = boto3.client('bedrock-runtime', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
courses_table = dynamodb.Table(COURSES_TABLE)
 
# OpenSearch auth (uses Lambda's role automatically)
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    'aoss',
    session_token=credentials.token
)
 
opensearch = OpenSearch(
    hosts=[{'host': OPENSEARCH_ENDPOINT.replace('https://', ''), 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30
)
 
 
# ───────────────────────────────────────────────────────────
# INTENT CLASSIFICATION
# ───────────────────────────────────────────────────────────
 
def classify_intent(question):
    """Ask Claude to classify the question's intent."""
    classification_prompt = (
        "Classify the following question into ONE category. Respond with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "- course_search: Looking for courses matching criteria (e.g., 'humanities classes at 11:15', '3-credit math courses', 'easy DSHU classes')\n"
        "- course_info: Asking about a specific course code (e.g., 'what does CMSC131 cover?', 'tell me about ENGL101')\n"
        "- professor_info: Asking about a specific professor (e.g., 'is Dr. Smith good?', 'what's Mount's rating?')\n"
        "- general: Anything else about UMD (admissions, dining, campus life, applying, majors overview)\n\n"
        f"Question: {question}\n\n"
        "Category:"
    )
 
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-6',
        body=json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 20,
            'temperature': 0,
            'messages': [{'role': 'user', 'content': classification_prompt}]
        })
    )
    result = json.loads(response['body'].read())
    raw = result['content'][0]['text'].strip().lower()
 
    if 'course_search' in raw:
        return 'course_search'
    if 'course_info' in raw:
        return 'course_info'
    if 'professor_info' in raw:
        return 'professor_info'
    return 'general'
 
 
# ───────────────────────────────────────────────────────────
# OPENSEARCH (general RAG)
# ───────────────────────────────────────────────────────────
 
def get_embedding(text):
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps({'inputText': text})
    )
    return json.loads(response['body'].read())['embedding']
 
 
def search_opensearch(question):
    embedding = get_embedding(question)
    query = {
        'size': TOP_K,
        'query': {'knn': {'embedding': {'vector': embedding, 'k': TOP_K}}},
        '_source': ['text', 'source']
    }
    response = opensearch.search(index=INDEX_NAME, body=query)
    return [hit['_source'] for hit in response['hits']['hits']]
 
 
# ───────────────────────────────────────────────────────────
# PLANETTERP API
# ───────────────────────────────────────────────────────────
 
def fetch_planetterp_professor(name):
    """Get prof info: rating, type, courses taught."""
    try:
        response = requests.get(
            f'{PLANETTERP_BASE}/professor',
            params={'name': name, 'reviews': 'false'},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"PlanetTerp error for {name}: {e}")
        return None
 
 
def fetch_planetterp_grades(professor_name, course_id):
    """Get grade distribution for a prof+course combo."""
    try:
        response = requests.get(
            f'{PLANETTERP_BASE}/grades',
            params={'professor': professor_name, 'course': course_id},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"PlanetTerp grades error for {professor_name}/{course_id}: {e}")
        return None
 
 
def calculate_pass_rate(grade_records):
    """From a list of grade records, calculate % of students who got C- or higher."""
    if not grade_records:
        return None
 
    total_students = 0
    passing_students = 0
    passing_grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-']
 
    for record in grade_records:
        for grade in passing_grades:
            total_students += record.get(grade, 0)
            passing_students += record.get(grade, 0)
        for grade in ['D+', 'D', 'D-', 'F', 'W']:
            total_students += record.get(grade, 0)
 
    if total_students == 0:
        return None
    return round((passing_students / total_students) * 100, 1)
 
 
def enrich_courses_with_professor_data(courses):
    """For each course, look up its instructors on PlanetTerp in parallel."""
    # Collect unique prof+course pairs to fetch
    lookups = []
    for course in courses:
        course_id = course.get('course_id')
        for section in course.get('sections', [])[:2]:  # First 2 sections only
            for instructor in section.get('instructors', [])[:1]:  # Primary instructor
                lookups.append((instructor, course_id))
 
    # Cap to keep total latency reasonable
    lookups = lookups[:MAX_PROFS_TO_ENRICH]
 
    # Parallel fetch
    enrichments = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        prof_futures = {
            executor.submit(fetch_planetterp_professor, prof): (prof, course_id)
            for prof, course_id in lookups
        }
        grade_futures = {
            executor.submit(fetch_planetterp_grades, prof, course_id): (prof, course_id)
            for prof, course_id in lookups
        }
 
        for future in prof_futures:
            prof, course_id = prof_futures[future]
            prof_data = future.result()
            if prof_data:
                enrichments[(prof, course_id)] = {
                    'rating': prof_data.get('average_rating'),
                    'type': prof_data.get('type')
                }
 
        for future in grade_futures:
            prof, course_id = grade_futures[future]
            grades = future.result()
            if grades:
                pass_rate = calculate_pass_rate(grades)
                if (prof, course_id) in enrichments:
                    enrichments[(prof, course_id)]['pass_rate'] = pass_rate
 
    return enrichments
 
 
# ───────────────────────────────────────────────────────────
# DYNAMODB (structured course search)
# ───────────────────────────────────────────────────────────
 
GEN_ED_MAP = {
    'humanities': 'DSHU',
    'history': 'DSHS',
    'social science': 'DSHS',
    'natural science': 'DSNS',
    'lab science': 'DSNL',
    'scholarship in practice': 'DSSP',
    'analytic reasoning': 'FSAR',
    'math': 'FSMA',
    'oral communication': 'FSOC',
    'professional writing': 'FSPW',
    'cultural competency': 'DVCC',
    'understanding plural societies': 'DVUP',
    'i-series': 'SCIS',
    'big questions': 'SCIS',
}
 
 
def extract_course_codes(question):
    return re.findall(r'\b([A-Z]{3,4}\d{3}[A-Z]?)\b', question.upper())
 
 
def extract_gen_eds(question):
    q_lower = question.lower()
    return [code for keyword, code in GEN_ED_MAP.items() if keyword in q_lower]
 
 
def extract_time(question):
    return re.findall(r'\b(\d{1,2}:\d{2})\b', question)
 
 
def extract_credits(question):
    match = re.search(r'\b(\d)\s*[- ]?credit', question.lower())
    return match.group(1) if match else None
 
 
def query_courses_by_code(codes):
    results = []
    for code in codes:
        try:
            response = courses_table.get_item(Key={'course_id': code})
            if 'Item' in response:
                results.append(response['Item'])
        except Exception as e:
            print(f"DynamoDB error for {code}: {e}")
    return results
 
 
def query_courses_by_filters(gen_eds, time_filter, credit_filter):
    filter_expression = None
 
    if gen_eds:
        ge_filter = Attr('gen_ed').contains(gen_eds[0])
        for ge in gen_eds[1:]:
            ge_filter = ge_filter | Attr('gen_ed').contains(ge)
        filter_expression = ge_filter
 
    if credit_filter:
        credit_attr = Attr('credits').eq(credit_filter)
        filter_expression = credit_attr if filter_expression is None else filter_expression & credit_attr
 
    if filter_expression is None:
        return []
 
    response = courses_table.scan(
        FilterExpression=filter_expression,
        Limit=300
    )
    items = response.get('Items', [])
 
    if time_filter:
        filtered = []
        for course in items:
            for section in course.get('sections', []):
                for meeting in section.get('meetings', []):
                    start = meeting.get('start_time', '')
                    if any(t in start for t in time_filter):
                        filtered.append(course)
                        break
                else:
                    continue
                break
        items = filtered
 
    items = [c for c in items if c.get('sections')]
    return items[:10]
 
 
def format_courses_for_prompt(courses, enrichments=None):
    """Compact, readable representation for Claude with optional prof enrichment."""
    if not courses:
        return ""
    lines = []
    for c in courses:
        course_id = c.get('course_id')
        section_info = []
        for s in c.get('sections', [])[:2]:
            instructors = s.get('instructors', [])
            instructor_str = ', '.join(instructors)
 
            # Add PlanetTerp data if available
            prof_extras = []
            if enrichments and instructors:
                key = (instructors[0], course_id)
                if key in enrichments:
                    e = enrichments[key]
                    if e.get('rating'):
                        prof_extras.append(f"PT rating: {e['rating']:.2f}/5")
                    if e.get('pass_rate'):
                        prof_extras.append(f"pass rate: {e['pass_rate']}%")
 
            extras_str = f" [{'; '.join(prof_extras)}]" if prof_extras else ""
 
            meetings = s.get('meetings', [])
            if meetings:
                m = meetings[0]
                section_info.append(
                    f"  Section {s.get('section_id')}: {instructor_str}{extras_str} | {m.get('days', '')} {m.get('start_time', '')}-{m.get('end_time', '')}"
                )
        section_text = '\n'.join(section_info) if section_info else '  (no sections offered)'
        lines.append(
            f"{course_id} - {c.get('name')} ({c.get('credits')} credits)\n"
            f"Gen-Ed: {', '.join(c.get('gen_ed', []) or ['none'])}\n"
            f"{section_text}"
        )
    return '\n\n'.join(lines)
 
 
# ───────────────────────────────────────────────────────────
# MAIN HANDLER
# ───────────────────────────────────────────────────────────
 
def handler(event, context):
    body = json.loads(event.get('body', '{}'))
    user_message = body.get('message', '')
 
    if len(user_message) > 200:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Message exceeds 200 characters'})
        }
    if not user_message:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No message provided'})
        }
 
    intent = classify_intent(user_message)
    print(f"Intent classified as: {intent}")
 
    context_text = ""
    used_source = "none"
    enrichments = {}
 
    if intent == 'course_search':
        gen_eds = extract_gen_eds(user_message)
        times = extract_time(user_message)
        credits = extract_credits(user_message)
        courses = query_courses_by_filters(gen_eds, times, credits)
        if courses:
            # Enrich with PlanetTerp data if the user asked about profs/passing
            wants_prof_data = any(
                k in user_message.lower()
                for k in ['pass', 'easy', 'hard', 'good', 'best', 'teacher', 'professor', 'rating']
            )
            if wants_prof_data:
                print("Enriching with PlanetTerp data...")
                enrichments = enrich_courses_with_professor_data(courses)
            context_text = "Matching courses from UMD catalog:\n\n" + format_courses_for_prompt(courses, enrichments)
            used_source = "dynamodb+planetterp" if enrichments else "dynamodb"
 
    elif intent == 'course_info':
        codes = extract_course_codes(user_message)
        if codes:
            courses = query_courses_by_code(codes)
            if courses:
                context_text = "Course details from UMD catalog:\n\n" + format_courses_for_prompt(courses)
                used_source = "dynamodb"
 
    elif intent == 'professor_info':
        # Extract prof name — heuristic: capitalized words after "Dr." or quoted names
        name_match = re.search(r'(?:Dr\.|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', user_message)
        if not name_match:
            # Fallback: try last capitalized word
            words = re.findall(r'\b([A-Z][a-z]+)\b', user_message)
            name_match_str = words[-1] if words else None
        else:
            name_match_str = name_match.group(1)
 
        if name_match_str:
            prof_data = fetch_planetterp_professor(name_match_str)
            if prof_data:
                context_text = f"Professor data from PlanetTerp:\n{json.dumps(prof_data, indent=2)}"
                used_source = "planetterp"
 
    else:  # general
        chunks = search_opensearch(user_message)
        if chunks:
            context_text = "Relevant UMD information:\n\n" + "\n\n".join(
                f"[{c['source']}]\n{c['text']}" for c in chunks
            )
            used_source = "opensearch"
 
    # Cascaded fallback
    if not context_text:
        print(f"Primary source empty for intent={intent}, falling back...")
        chunks = search_opensearch(user_message)
        if chunks:
            context_text = "Related UMD information:\n\n" + "\n\n".join(
                f"[{c['source']}]\n{c['text']}" for c in chunks
            )
            used_source = "opensearch (fallback)"
 
    print(f"Used source: {used_source}")
 
    system_prompt = (
        "You are a helpful assistant for University of Maryland students. "
        "Answer the user's question using the context below. "
        "When professor pass rates or ratings are provided, use them to inform your recommendations. "
        "If the context doesn't contain enough information, say so honestly. "
        "Be concise and direct.\n\n"
        f"Context:\n{context_text if context_text else '(no relevant information found)'}"
    )
 
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-6',
        body=json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1024,
            'temperature': 0.7,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_message}]
        })
    )
 
    result = json.loads(response['body'].read())
    reply = result['content'][0]['text']
 
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({'reply': reply, 'intent': intent, 'source': used_source})
    }