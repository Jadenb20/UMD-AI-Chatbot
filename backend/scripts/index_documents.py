import boto3

import json

import os

from opensearchpy import OpenSearch, RequestsHttpConnection

from requests_aws4auth import AWS4Auth

import subprocess
 
# Configuration

result = subprocess.run(
    ['terraform', '-chdir=../../terraform', 'output', '-raw', 'opensearch_endpoint'],
    capture_output=True, text=True, check=True
)
OPENSEARCH_ENDPOINT = result.stdout.strip()
print(f"Using endpoint: {OPENSEARCH_ENDPOINT}")

REGION = 'us-east-1'
INDEX_NAME = 'umd-knowledge'
DOCS_DIR = '../knowledge_base/docs'
 
# AWS clients

bedrock = boto3.client('bedrock-runtime', region_name=REGION)

session = boto3.Session()

credentials = session.get_credentials()

awsauth = AWS4Auth(

    credentials.access_key,

    credentials.secret_key,

    REGION,

    'aoss',

    session_token=credentials.token

)
 
# OpenSearch client

opensearch = OpenSearch(

    hosts=[{'host': OPENSEARCH_ENDPOINT.replace('https://', ''), 'port': 443}],

    http_auth=awsauth,

    use_ssl=True,

    verify_certs=True,

    connection_class=RequestsHttpConnection,
    
    timeout= 30

)
 
def chunk_text(text, chunk_size=500, overlap=50):

    """Split text into overlapping chunks."""

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size - overlap):

        chunk = ' '.join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks
 
def generate_embedding(text):

    """Generate embedding using Bedrock Titan."""

    response = bedrock.invoke_model(

        modelId='amazon.titan-embed-text-v1',

        body=json.dumps({'inputText': text})

    )

    result = json.loads(response['body'].read())

    return result['embedding']
 
def create_index():

    """Create OpenSearch index with vector field."""

    index_body = {

        'settings': {

            'index': {

                'knn': True,

                'knn.algo_param.ef_search': 512

            }

        },

        'mappings': {

            'properties': {

                'text': {'type': 'text'},

                'embedding': {

                    'type': 'knn_vector',

                    'dimension': 1536,

                    'method': {

                        'name': 'hnsw',

                        'space_type': 'l2',

                        'engine': 'nmslib'

                    }

                },

                'source': {'type': 'keyword'}

            }

        }

    }

    if not opensearch.indices.exists(index=INDEX_NAME):

        opensearch.indices.create(index=INDEX_NAME, body=index_body)

        print(f"Created index: {INDEX_NAME}")
 
def index_documents():

    """Read documents, chunk them, generate embeddings, and index."""

    create_index()

    doc_count = 0

    for filename in os.listdir(DOCS_DIR):

        if not filename.endswith('.txt'):

            continue

        filepath = os.path.join(DOCS_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:

            text = f.read()

        chunks = chunk_text(text)

        print(f"Processing {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):

            embedding = generate_embedding(chunk)

            doc = {

                'text': chunk,

                'embedding': embedding,

                'source': filename

            }

            opensearch.index(index=INDEX_NAME, body=doc)

            doc_count += 1

            if doc_count % 10 == 0:

                print(f"  Indexed {doc_count} chunks...")

    print(f"Done! Indexed {doc_count} total chunks.")
 
if __name__ == '__main__':

    index_documents()
 