import boto3

# Configuration
REGION = 'us-east-1'
COURSES_TABLE = 'umd-chatbot-courses'
INSTRUCTOR_INDEX_TABLE = 'umd-chatbot-instructor-index'

dynamodb = boto3.resource('dynamodb', region_name=REGION)
courses_table = dynamodb.Table(COURSES_TABLE)
instructor_index_table = dynamodb.Table(INSTRUCTOR_INDEX_TABLE)


def normalize_instructor_name(name):
    return name.strip().lower()


def scan_all_courses():
    """Paginate through every item in the courses table."""
    items = []
    scan_kwargs = {}
    while True:
        response = courses_table.scan(**scan_kwargs)
        items.extend(response.get('Items', []))
        last_key = response.get('LastEvaluatedKey')
        if not last_key:
            break
        scan_kwargs['ExclusiveStartKey'] = last_key
    return items


def build_instructor_index_items(courses):
    """Flatten each course's sections[].instructors[] into one
    (instructor_name, course_id) pair per instructor per course."""
    seen = set()
    items = []
    for course in courses:
        course_id = course.get('course_id')
        for section in course.get('sections', []):
            for instructor in section.get('instructors', []):
                key = (normalize_instructor_name(instructor), course_id)
                if key not in seen:
                    seen.add(key)
                    items.append({'instructor_name': key[0], 'course_id': key[1]})
    return items


def load_to_dynamodb(items):
    with instructor_index_table.batch_writer() as batch:
        for i, item in enumerate(items):
            batch.put_item(Item=item)
            if (i + 1) % 50 == 0:
                print(f'  Loaded {i + 1} instructor-index rows...')
    print(f'Done! Loaded {len(items)} instructor-index rows.')


def main():
    print('Scanning courses table...')
    courses = scan_all_courses()
    print(f'Scanned {len(courses)} courses.')

    items = build_instructor_index_items(courses)
    print(f'Built {len(items)} unique (instructor, course) pairs.')

    load_to_dynamodb(items)


if __name__ == '__main__':
    main()
