import json
import boto3



bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def handler(event, context):
    body = json.loads(event.get('body', '{}'))
    user_message = body.get('message', '')
    
    max_length = 100  # Set your desired maximum length
 
    if len(user_message) > max_length:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Message exceeds maximum length of {max_length} characters'})
        }

    if not user_message:
        return {'statusCode': 400, 'body': json.dumps({'error': 'No message provided'})}

    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-6',
        body=json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1024,
            'temperature': 0.7,
            'system': 'You are a helpful UMD assistant.',
            'messages': [{'role': 'user', 'content': user_message}]
        })
    )

    result = json.loads(response['body'].read())
    reply = result['content'][0]['text']

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}, #too much access
        'body': json.dumps({'reply': reply})
    }