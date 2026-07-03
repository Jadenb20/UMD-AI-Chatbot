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
MAX_PROFS_TO_ENRICH = 20
 
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
# INTENT CLASSIFICATION (history-aware, sharper parsing)
# ───────────────────────────────────────────────────────────
 
def classify_intent(question, history=None):
    """Ask Claude to classify the question's intent, using recent conversation context."""
 
    context_blurb = ""
    if history:
        recent = history[-4:] if len(history) > 4 else history
        context_blurb = "Recent conversation:\n"
        for msg in recent:
            role = msg.get('role', '')
            content = msg.get('content', '')[:200]
            context_blurb += f"{role}: {content}\n"
        context_blurb += "\n"
 
    classification_prompt = (
        "Classify the following question into ONE category. Reply with ONLY the category name — no other words, no explanation.\n\n"
        "IMPORTANT RULES:\n"
        "- If a specific course code appears (like CMSC131, ENGL101), OR the question refers to a course from recent conversation, classify as course_info.\n"
        "- 'Who teaches X', 'who has the highest rating', 'best professor for X', 'which prof is easiest' — these are course_info when a course context exists.\n"
        "- Only use professor_info when the question is about ONE named professor and NOT comparing across a course.\n"
        "- If the question refers to a professor already mentioned in recent conversation (using 'her', 'him', 'they', 'their class'), classify as professor_info.\n\n"
        "Valid category names (choose exactly one):\n"
        "course_search — filtering courses by criteria (times, gen-eds, credits) with no specific course code\n"
        "course_info — about a specific course code OR ranking/comparing profs of a course\n"
        "professor_info — about ONE named professor (or pronoun referring to one)\n"
        "general — anything else about UMD (admissions, dining, campus life)\n\n"
        f"{context_blurb}"
        f"Current question: {question}\n\n"
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
 
    # Extract just the category name — grab the first whitespace-delimited token
    # This handles cases where Claude adds explanations after the category
    first_token = raw.split()[0].strip('.,;:') if raw.split() else ''
 
    if first_token == 'course_search' or first_token.startswith('course_search'):
        return 'course_search'
    if first_token == 'course_info' or first_token.startswith('course_info'):
        return 'course_info'
    if first_token == 'professor_info' or first_token.startswith('professor_info'):
        return 'professor_info'
    if first_token == 'general':
        return 'general'
 
    # Fallback: check whole response for any category as a whole word
    for category in ['course_search', 'course_info', 'professor_info']:
        if re.search(rf'\b{category}\b', raw):
            return category
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
# PLANETTERP API (with name-variant fallback)
# ───────────────────────────────────────────────────────────
 
def _try_planetterp_professor(name):
    """One-shot lookup, returns None on 404 or error."""
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
 
 
def fetch_planetterp_professor(name):
    """Try name variants to work around middle name / initial mismatches."""
    if not name:
        return None
 
    # Try the name as-is first
    data = _try_planetterp_professor(name)
    if data:
        return data
 
    # Try common variants
    parts = name.split()
    variants = []
 
    if len(parts) >= 2:
        # Just first + last (drop middle names/initials)
        variants.append(f"{parts[0]} {parts[-1]}")
        # Last, First (some listings use this)
        variants.append(f"{parts[-1]}, {parts[0]}")
 
    for variant in variants:
        if variant == name:
            continue
        data = _try_planetterp_professor(variant)
        if data:
            print(f"Found {name} via variant: {variant}")
            return data
 
    return None
 
 
def fetch_planetterp_grades(professor_name, course_id):
    try:
        params = {'professor': professor_name}
        if course_id:
            params['course'] = course_id
        response = requests.get(f'{PLANETTERP_BASE}/grades', params=params, timeout=5)
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
 
 
def enrich_courses_with_professor_data(courses, cap=MAX_PROFS_TO_ENRICH):
    """For each course, look up its instructors on PlanetTerp in parallel."""
    lookups = []
    seen = set()
    for course in courses:
        course_id = course.get('course_id')
        for section in course.get('sections', []):
            for instructor in section.get('instructors', []):
                key = (instructor, course_id)
                if key not in seen:
                    seen.add(key)
                    lookups.append(key)
 
    lookups = lookups[:cap]
 
    enrichments = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
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
# EXTRACTION HELPERS
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
    """Extract course codes like CMSC131, ENGL101, PHIL220 — case-insensitive."""
    return re.findall(r'\b([A-Za-z]{3,4}\d{3}[A-Za-z]?)\b', question.upper())
 
 
def extract_gen_eds(question):
    q_lower = question.lower()
    return [code for keyword, code in GEN_ED_MAP.items() if keyword in q_lower]
 
 
def extract_time(question):
    """Extract time filters like '11:15', '11am', '11 am', '2:30pm'."""
    times = []
    times.extend(re.findall(r'\b(\d{1,2}:\d{2})\b', question))
    for match in re.finditer(r'\b(\d{1,2})\s*(am|pm)\b', question.lower()):
        hour = match.group(1)
        times.append(f"{hour}:00{match.group(2)}")
    return times
 
 
def extract_credits(question):
    match = re.search(r'\b(\d)\s*[- ]?credit', question.lower())
    return match.group(1) if match else None
 
 
def find_professor_name_in_history(history):
    """Scan recent messages backwards for a two-word capitalized name."""
    if not history:
        return None
    for msg in reversed(history):
        content = msg.get('content', '')
        # Look for "First Last" pattern (two capitalized words)
        match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', content)
        if match:
            return match.group(1)
    return None
 
 
def find_course_code_in_history(history):
    """Scan recent messages backwards for a course code — case-insensitive."""
    if not history:
        return None
    for msg in reversed(history):
        content = msg.get('content', '')
        match = re.search(r'\b([A-Za-z]{3,4}\d{3}[A-Za-z]?)\b', content)
        if match:
            return match.group(1).upper()
    return None
 
 
# ───────────────────────────────────────────────────────────
# DYNAMODB (structured course search)
# ───────────────────────────────────────────────────────────
 
def query_courses_by_code(codes):
    results = []
    for code in codes:
        try:
            response = courses_table.get_item(Key={'course_id': code.upper()})
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
    if not courses:
        return ""
    lines = []
    for c in courses:
        course_id = c.get('course_id')
        section_info = []
        for s in c.get('sections', []):
            instructors = s.get('instructors', [])
            instructor_str = ', '.join(instructors)
 
            prof_extras = []
            if enrichments and instructors:
                key = (instructors[0], course_id)
                if key in enrichments:
                    e = enrichments[key]
                    if e.get('rating'):
                        prof_extras.append(f"PT rating: {e['rating']:.2f}/5")
                    else:
                        prof_extras.append("no rating on PlanetTerp")
                    if e.get('pass_rate') is not None:
                        prof_extras.append(f"pass rate: {e['pass_rate']}%")
                else:
                    prof_extras.append("no PlanetTerp data")
 
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
 
 
PROF_QUALITY_KEYWORDS = [
    'pass', 'easy', 'hard', 'good', 'best', 'teacher', 'professor',
    'rating', 'highest', 'lowest', 'which', 'recommend', 'chance', 'compare'
]
 
 
def wants_prof_data(question):
    return any(k in question.lower() for k in PROF_QUALITY_KEYWORDS)
 
 
# ───────────────────────────────────────────────────────────
# MAIN HANDLER
# ───────────────────────────────────────────────────────────
 
def handler(event, context):
    body = json.loads(event.get('body', '{}'))
    user_message = body.get('message', '')
    history = body.get('history', [])
 
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
 
    intent = classify_intent(user_message, history)
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
            if wants_prof_data(user_message):
                print("Enriching with PlanetTerp data...")
                enrichments = enrich_courses_with_professor_data(courses)
            context_text = "Matching courses from UMD catalog:\n\n" + format_courses_for_prompt(courses, enrichments)
            used_source = "dynamodb+planetterp" if enrichments else "dynamodb"
 
    elif intent == 'course_info':
        codes = extract_course_codes(user_message)
        # Fallback: check conversation history
        if not codes:
            historical_code = find_course_code_in_history(history)
            if historical_code:
                codes = [historical_code]
                print(f"Using course code from history: {historical_code}")
 
        if codes:
            courses = query_courses_by_code(codes)
            if courses:
                if wants_prof_data(user_message):
                    print("Enriching course_info with PlanetTerp data...")
                    # Higher cap for single-course lookups — we want all profs
                    enrichments = enrich_courses_with_professor_data(courses, cap=40)
                context_text = "Course details from UMD catalog:\n\n" + format_courses_for_prompt(courses, enrichments)
                used_source = "dynamodb+planetterp" if enrichments else "dynamodb"
 
    elif intent == 'professor_info':
        name_match = re.search(r'(?:Dr\.|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', user_message)
        if name_match:
            name_match_str = name_match.group(1)
        else:
            two_word = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', user_message)
            name_match_str = two_word.group(1) if two_word else None
 
        # Fallback: check history
        if not name_match_str:
            name_match_str = find_professor_name_in_history(history)
            if name_match_str:
                print(f"Using professor name from history: {name_match_str}")
 
        if name_match_str:
            prof_data = fetch_planetterp_professor(name_match_str)
            if prof_data:
                grade_records = fetch_planetterp_grades(name_match_str, None)
                pass_rate = calculate_pass_rate(grade_records) if grade_records else None
 
                summary_parts = [f"Professor: {name_match_str}"]
                if prof_data.get('average_rating'):
                    summary_parts.append(f"Average rating: {prof_data['average_rating']:.2f}/5")
                if prof_data.get('type'):
                    summary_parts.append(f"Type: {prof_data['type']}")
                if pass_rate is not None:
                    summary_parts.append(f"Overall pass rate (C- or higher): {pass_rate}%")
                if prof_data.get('courses'):
                    summary_parts.append(f"Courses taught: {', '.join(prof_data['courses'][:10])}")
 
                context_text = "Professor data from PlanetTerp:\n" + "\n".join(summary_parts)
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
        "When professor pass rates, ratings, or grade data are provided, prominently cite the specific numbers and use them to make concrete recommendations. "
        "If the user asks 'which' or 'best' or 'highest' — provide a clear ranking. "
        "If the context shows 'no PlanetTerp data' or 'no rating on PlanetTerp' for some professors, acknowledge them but focus recommendations on those with data. "
        "If the context doesn't contain enough information, say so honestly and briefly suggest checking PlanetTerp or Testudo. "
        "Be concise and direct.\n\n"
        f"Context:\n{context_text if context_text else '(no relevant information found)'}"
    )
 
    # Build conversation history for Claude
    conv_history = history[:]
    while conv_history and conv_history[0].get('role') == 'assistant':
        conv_history = conv_history[1:]
    conv_history = conv_history[-10:] if len(conv_history) > 10 else conv_history
 
    messages = conv_history + [{'role': 'user', 'content': user_message}]
 
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-6',
        body=json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1024,
            'temperature': 0.7,
            'system': system_prompt,
            'messages': messages
        })
    )
 
    result = json.loads(response['body'].read())
    reply = result['content'][0]['text']
 
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({'reply': reply, 'intent': intent, 'source': used_source})
    }