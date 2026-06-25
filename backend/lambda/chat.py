import json
import boto3
import os
import re
from boto3.dynamodb.conditions import Attr
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
 
# Configuration
REGION = 'us-east-1'
OPENSEARCH_ENDPOINT = os.environ['OPENSEARCH_ENDPOINT']
INDEX_NAME = 'umd-knowledge'
COURSES_TABLE = 'umd-chatbot-courses'
TOP_K = 3
 
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
        "- course_search: Looking for courses matching criteria (e.g., 'humanities classes at 11:15', '3-credit math courses')\n"
        "- course_info: Asking about a specific course code (e.g., 'what does CMSC131 cover?', 'tell me about ENGL101')\n"
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
 
    # Normalize the response
    if 'course_search' in raw:
        return 'course_search'
    if 'course_info' in raw:
        return 'course_info'
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
# DYNAMODB (structured course search)
# ───────────────────────────────────────────────────────────
 
# Maps human words → UMD gen-ed codes
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
    """Pull course codes like CMSC131, ENGL101 from the question."""
    return re.findall(r'\b([A-Z]{3,4}\d{3}[A-Z]?)\b', question.upper())
 
 
def extract_gen_eds(question):
    """Find any gen-ed keywords in the question."""
    q_lower = question.lower()
    return [code for keyword, code in GEN_ED_MAP.items() if keyword in q_lower]
 
 
def extract_time(question):
    """Find times like 11:15, 2:30, etc."""
    return re.findall(r'\b(\d{1,2}:\d{2})\b', question)
 
 
def extract_credits(question):
    """Find credit hours like '3 credit' or '3-credit'."""
    match = re.search(r'\b(\d)\s*[- ]?credit', question.lower())
    return match.group(1) if match else None
 
 
def query_courses_by_code(codes):
    """Look up specific courses by their IDs."""
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
    """Scan DynamoDB with filters. Limited to reasonable result counts."""
    filter_expression = None
 
    if gen_eds:
        # Match any of the requested gen-eds
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
        Limit=300  # Cap raw scan
    )
    items = response.get('Items', [])
 
    # Now do time filtering in Python (DynamoDB can't filter on nested arrays well)
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
 
    # Skip courses with no sections (the noise from the indexer)
    items = [c for c in items if c.get('sections')]
 
    return items[:10]  # Return top 10 to keep prompt manageable
 
 
def format_courses_for_prompt(courses):
    """Compact, readable representation for Claude."""
    if not courses:
        return ""
    lines = []
    for c in courses:
        section_info = []
        for s in c.get('sections', [])[:2]:  # First 2 sections only
            instructors = ', '.join(s.get('instructors', []))
            meetings = s.get('meetings', [])
            if meetings:
                m = meetings[0]
                section_info.append(f"  Section {s.get('section_id')}: {instructors} | {m.get('days', '')} {m.get('start_time', '')}-{m.get('end_time', '')}")
        section_text = '\n'.join(section_info) if section_info else '  (no sections offered)'
        lines.append(
            f"{c.get('course_id')} - {c.get('name')} ({c.get('credits')} credits)\n"
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
 
    # ── Step 1: Classify intent ──
    intent = classify_intent(user_message)
    print(f"Intent classified as: {intent}")
 
    # ── Step 2: Primary retrieval based on intent ──
    context_text = ""
    used_source = "none"
 
    if intent == 'course_search':
        # Extract filters and query DynamoDB
        gen_eds = extract_gen_eds(user_message)
        times = extract_time(user_message)
        credits = extract_credits(user_message)
        courses = query_courses_by_filters(gen_eds, times, credits)
        if courses:
            context_text = "Matching courses from UMD catalog:\n\n" + format_courses_for_prompt(courses)
            used_source = "dynamodb"
 
    elif intent == 'course_info':
        # Look up specific course codes
        codes = extract_course_codes(user_message)
        if codes:
            courses = query_courses_by_code(codes)
            if courses:
                context_text = "Course details from UMD catalog:\n\n" + format_courses_for_prompt(courses)
                used_source = "dynamodb"
 
    else:  # general
        chunks = search_opensearch(user_message)
        if chunks:
            context_text = "Relevant UMD information:\n\n" + "\n\n".join(
                f"[{c['source']}]\n{c['text']}" for c in chunks
            )
            used_source = "opensearch"
 
    # ── Step 3: Fallback if primary returned nothing ──
    if not context_text:
        print(f"Primary source empty for intent={intent}, falling back...")
        if intent in ('course_search', 'course_info'):
            chunks = search_opensearch(user_message)
            if chunks:
                context_text = "Related UMD information:\n\n" + "\n\n".join(
                    f"[{c['source']}]\n{c['text']}" for c in chunks
                )
                used_source = "opensearch (fallback)"
        else:
            codes = extract_course_codes(user_message)
            if codes:
                courses = query_courses_by_code(codes)
                if courses:
                    context_text = "Course details from UMD catalog:\n\n" + format_courses_for_prompt(courses)
                    used_source = "dynamodb (fallback)"
 
    print(f"Used source: {used_source}")
 
    # ── Step 4: Ask Claude to answer with whatever context we got ──
    system_prompt = (
        "You are a helpful assistant for University of Maryland students. "
        "Answer the user's question using the context below. "
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