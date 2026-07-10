import boto3
import requests
import subprocess
import time
 
# Configuration
REGION = 'us-east-1'
SEMESTER = '202608'  # Fall 2026 — adjust as needed
TABLE_NAME = 'umd-chatbot-courses'
INSTRUCTOR_INDEX_TABLE_NAME = 'umd-chatbot-instructor-index'

# umd.io endpoints
UMDIO_BASE = 'https://api.umd.io/v1'

# DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table(TABLE_NAME)
instructor_index_table = dynamodb.Table(INSTRUCTOR_INDEX_TABLE_NAME)
 
 
def fetch_all_courses():

    """Fetch every course from umd.io, paginating through results."""

    courses = []

    page = 1

    per_page = 100
 
    while True:

        url = f'{UMDIO_BASE}/courses'

        params = {'semester': SEMESTER, 'page': page, 'per_page': per_page}

        print(f'Fetching page {page}...')
 
        # Retry up to 3 times on network hiccups

        for attempt in range(3):

            try:

                response = requests.get(url, params=params, timeout=30)

                response.raise_for_status()

                batch = response.json()

                break

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:

                if attempt < 2:

                    print(f'  ⚠ Timeout on page {page}, retrying ({attempt + 1}/3)...')

                    time.sleep(2)

                else:

                    print(f'  ✗ Page {page} failed after 3 attempts, giving up: {e}')

                    return courses  # return what we have so far
 
        if not batch:

            break

        courses.extend(batch)

        if len(batch) < per_page:

            break

        page += 1

        time.sleep(0.2)
 
    print(f'Fetched {len(courses)} courses total.')

    return courses
 
 
 
def fetch_sections(course_id):
    """Fetch all sections (times, professors, rooms) for one course."""
    url = f'{UMDIO_BASE}/courses/{course_id}/sections'
    try:
        response = requests.get(url, params={'semester': SEMESTER}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'  ⚠ Failed to fetch sections for {course_id}: {e}')
        return []
 
 
def flatten_gen_ed(raw_gen_ed):
    """
    umd.io returns gen_ed as nested lists like [["FSAW"], ["DSHU","DSHS"]]
    representing AND/OR grouping. We flatten to a simple list of codes
    so DynamoDB's contains() filter can match them directly.
    """
    flat = []
    for group in raw_gen_ed or []:
        if isinstance(group, list):
            flat.extend(group)
        else:
            flat.append(group)
    return flat
 
 
def build_course_item(course, sections):
    """Shape one course + its sections into a DynamoDB item."""
    item = {
        'course_id': course['course_id'],
        'name': course.get('name', ''),
        'dept_id': course.get('dept_id', ''),
        'credits': course.get('credits', ''),
        'description': course.get('description', '') or '',
        'gen_ed': flatten_gen_ed(course.get('gen_ed', [])),
        'sections': []
    }
 
    for s in sections:
        section_summary = {
            'section_id': s.get('section_id', ''),
            'instructors': s.get('instructors', []),
            'seats': s.get('seats', ''),
            'open_seats': s.get('open_seats', ''),
            'meetings': [
                {
                    'days': m.get('days', ''),
                    'start_time': m.get('start_time', ''),
                    'end_time': m.get('end_time', ''),
                    'room': m.get('room', ''),
                    'building': m.get('building', ''),
                    'classtype': m.get('classtype', '')
                }
                for m in s.get('meetings', [])
            ]
        }
        item['sections'].append(section_summary)
 
    return item
 
 
def load_to_dynamodb(items):
    """Batch-write items to DynamoDB (25 at a time, the API max)."""
    with table.batch_writer() as batch:
        for i, item in enumerate(items):
            batch.put_item(Item=item)
            if (i + 1) % 50 == 0:
                print(f'  Loaded {i + 1} courses...')
    print(f'Done! Loaded {len(items)} courses into DynamoDB.')


def normalize_instructor_name(name):
    return name.strip().lower()


def build_instructor_index_items(items):
    """Flatten each course's sections[].instructors[] into one
    (instructor_name, course_id) pair per instructor per course, so the
    instructor-index table stays in sync with what was just loaded above."""
    seen = set()
    index_items = []
    for course in items:
        course_id = course.get('course_id')
        for section in course.get('sections', []):
            for instructor in section.get('instructors', []):
                key = (normalize_instructor_name(instructor), course_id)
                if key not in seen:
                    seen.add(key)
                    index_items.append({'instructor_name': key[0], 'course_id': key[1]})
    return index_items


def load_instructor_index_to_dynamodb(index_items):
    with instructor_index_table.batch_writer() as batch:
        for i, item in enumerate(index_items):
            batch.put_item(Item=item)
            if (i + 1) % 50 == 0:
                print(f'  Loaded {i + 1} instructor-index rows...')
    print(f'Done! Loaded {len(index_items)} instructor-index rows into DynamoDB.')


def main():
    courses = fetch_all_courses()
    items = []

    for i, course in enumerate(courses):
        sections = fetch_sections(course['course_id'])
        if not sections:
            continue  # skip courses not actually offered this semester
        item = build_course_item(course, sections)
        items.append(item)

        if (i + 1) % 25 == 0:
            print(f'  Processed {i + 1}/{len(courses)} courses (with sections)...')
        time.sleep(0.1)  # rate-limit politeness

    load_to_dynamodb(items)
    load_instructor_index_to_dynamodb(build_instructor_index_items(items))
 
 
if __name__ == '__main__':
    main()
 