import boto3

import requests

import subprocess

import time
 
# Configuration

REGION = 'us-east-1'

SEMESTER = '202608'  # Fall 2026 — adjust as needed

TABLE_NAME = 'umd-chatbot-courses'
 
# umd.io endpoints

UMDIO_BASE = 'https://api.umd.io/v1'
 
# DynamoDB client

dynamodb = boto3.resource('dynamodb', region_name=REGION)

table = dynamodb.Table(TABLE_NAME)
 
 
def fetch_all_courses():

    """Fetch every course from umd.io, paginating through results."""

    courses = []

    page = 1

    per_page = 100
 
    while True:

        url = f'{UMDIO_BASE}/courses'

        params = {'semester': SEMESTER, 'page': page, 'per_page': per_page}

        print(f'Fetching page {page}...')

        response = requests.get(url, params=params, timeout=30)

        response.raise_for_status()

        batch = response.json()

        if not batch:

            break

        courses.extend(batch)

        if len(batch) < per_page:

            break

        page += 1

        time.sleep(0.2)  # be polite to umd.io
 
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
 
 
def build_course_item(course, sections):

    """Shape one course + its sections into a DynamoDB item."""

    item = {

        'course_id': course['course_id'],

        'name': course.get('name', ''),

        'dept_id': course.get('dept_id', ''),

        'credits': course.get('credits', ''),

        'description': course.get('description', '') or '',

        'gen_ed': course.get('gen_ed', []) or [],

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

                    'building': m.get('building', '')

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
 
 
def main():

    courses = fetch_all_courses()

    items = []
 
    for i, course in enumerate(courses):

        sections = fetch_sections(course['course_id'])

        item = build_course_item(course, sections)

        items.append(item)
 
        if (i + 1) % 25 == 0:

            print(f'  Processed {i + 1}/{len(courses)} courses (with sections)...')

        time.sleep(0.1)  # rate-limit politeness
 
    load_to_dynamodb(items)
 
 
if __name__ == '__main__':

    main()
 