# Set up S3 bucket

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "frontend" {
  bucket = "umd-chatbot-frontend-${random_id.suffix.hex}"
}


resource "aws_s3_bucket" "knowledge_base" {
  bucket = "umd-chatbot-kb-${random_id.suffix.hex}"
}

resource "random_id" "suffix" {
  byte_length = 4
}

#Lambda IAM Role

resource "aws_iam_role" "lambda_exec" {
  name = "umd-chatbot-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Lamda to call Bedrock

resource "aws_iam_role_policy" "lambda_bedrock" {
  name = "bedrock-and-opensearch"
  role = aws_iam_role.lambda_exec.id
 
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeBedrock"
        Effect = "Allow"
        Action = ["bedrock:GetFoundationModel", "bedrock:InvokeModel"]
        Resource = [
          "arn:aws:bedrock:us-east-1:${data.aws_caller_identity.current.account_id}:inference-profile/us.anthropic.claude-sonnet-4-6",
          "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6*",
          "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-sonnet-4-6*",
          "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-6*",
          "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
        ]
      },
      {
        Sid    = "AllowMarketplaceSubscribe"
        Effect = "Allow"
        Action = [
          "aws-marketplace:ViewSubscriptions",
          "aws-marketplace:Subscribe",
          "aws-marketplace:Unsubscribe"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:CalledViaLast" = "bedrock.amazonaws.com"
          }
        }
      },
      {
        Sid      = "AccessOpenSearch"
        Effect   = "Allow"
        Action   = ["aoss:APIAccessAll"]
        Resource = "*"
      },
      {
        Sid      = "ReadCoursesTable"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"]
        Resource = aws_dynamodb_table.courses.arn
      },
      {
        Sid      = "ReadInstructorIndexTable"
        Effect   = "Allow"
        Action   = ["dynamodb:Query"]
        Resource = aws_dynamodb_table.instructor_index.arn
      }
    ]
  })
}

#Lambda Logging

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}


data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../backend/lambda"
  output_path = "${path.module}/../backend/lambda_deployment.zip"
  excludes    = ["chat.zip", "__pycache__"]
}

# Lambda function
resource "aws_lambda_function" "chat" {
  function_name    = "umd-chatbot-chat"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  role             = aws_iam_role.lambda_exec.arn
  handler          = "chat.handler"
  runtime          = "python3.11"
  timeout          = 30
  memory_size      = 256
 
  environment {
    variables = {
      OPENSEARCH_ENDPOINT = aws_opensearchserverless_collection.knowledge_base.collection_endpoint
    }
  }
}

resource "aws_apigatewayv2_api" "chatbot" {
  name          = "umd-chatbot-api"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type"]
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id           = aws_apigatewayv2_api.chatbot.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.chat.invoke_arn
}

resource "aws_apigatewayv2_route" "chat" {
  api_id    = aws_apigatewayv2_api.chatbot.id
  route_key = "POST /chat"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.chatbot.id
  name        = "$default"
  auto_deploy = true
 
  # TEMPORARY: loosened for edge-case-tester agent testing against the live
  # deployed endpoint. Revert to production-safe values (rate_limit = 1,
  # burst_limit = 5, or whatever we determine is appropriate) once testing
  # is complete.
  default_route_settings {
    throttling_rate_limit  = 10  # ~10 req/sec sustained
    throttling_burst_limit = 25  # allows brief bursts up to 25
  }
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chat.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.chatbot.execution_arn}/*/*"
}

output "api_url" {
  value = aws_apigatewayv2_stage.default.invoke_url
}

# OpenSearch Serverless collection for RAG

resource "aws_opensearchserverless_security_policy" "encryption" {

  name = "umd-chatbot-encryption"

  type = "encryption"

  policy = jsonencode({

    Rules = [{

      ResourceType = "collection"

      Resource     = ["collection/umd-chatbot-knowledge"]

    }]

    AWSOwnedKey = true

  })

}
 
resource "aws_opensearchserverless_security_policy" "network" {

  name = "umd-chatbot-network"

  type = "network"

  policy = jsonencode([{

    Rules = [{

      ResourceType = "collection"

      Resource     = ["collection/umd-chatbot-knowledge"]

    }]

    AllowFromPublic = true

  }])

}
 
resource "aws_opensearchserverless_access_policy" "data_access" {
  name = "umd-chatbot-access"
  type = "data"
  policy = jsonencode([
    {
      Rules = [{
        ResourceType = "collection"
        Resource     = ["collection/umd-chatbot-knowledge"]
        Permission   = ["aoss:DescribeCollectionItems"]
      }, {
        ResourceType = "index"
        Resource     = ["index/umd-chatbot-knowledge/*"]
        Permission   = ["aoss:ReadDocument", "aoss:DescribeIndex"]
      }]
      Principal = [aws_iam_role.lambda_exec.arn]
    },
    {
      Rules = [{
        ResourceType = "collection"
        Resource     = ["collection/umd-chatbot-knowledge"]
        Permission   = ["aoss:*"]
      }, {
        ResourceType = "index"
        Resource     = ["index/umd-chatbot-knowledge/*"]
        Permission   = ["aoss:*"]
      }]
      Principal = ["arn:aws:iam::301394199680:user/jaden-admin"]
    }
  ])
}
 
resource "aws_opensearchserverless_collection" "knowledge_base" {

  name = "umd-chatbot-knowledge"

  type = "VECTORSEARCH"
 
  depends_on = [

    aws_opensearchserverless_security_policy.encryption,

    aws_opensearchserverless_security_policy.network,

    aws_opensearchserverless_access_policy.data_access

  ]

}
 
output "opensearch_endpoint" {

  value = aws_opensearchserverless_collection.knowledge_base.collection_endpoint

}
 

# DynamoDB table for structured course data
resource "aws_dynamodb_table" "courses" {
  name         = "umd-chatbot-courses"
  billing_mode = "PAY_PER_REQUEST"  # No idle cost, pay per query
  hash_key     = "course_id"

  attribute {
    name = "course_id"
    type = "S"
  }

  # Tag for easy identification
  tags = {
    Project = "umd-chatbot"
  }
}

output "courses_table_name" {
  value = aws_dynamodb_table.courses.name
}

# Lookup table for "which courses does this professor teach" queries.
# The courses table can't answer this directly — instructor names live inside
# a nested sections[].instructors[] list, which DynamoDB can't build a GSI on.
# This table is a flat (instructor_name, course_id) pair per row instead, kept
# in sync by index_courses.py / backfill_instructor_index.py.
resource "aws_dynamodb_table" "instructor_index" {
  name         = "umd-chatbot-instructor-index"
  billing_mode = "PAY_PER_REQUEST"  # No idle cost, pay per query
  hash_key     = "instructor_name"
  range_key    = "course_id"

  attribute {
    name = "instructor_name"  # normalized (trimmed, lowercased) for lookup consistency
    type = "S"
  }

  attribute {
    name = "course_id"
    type = "S"
  }

  tags = {
    Project = "umd-chatbot"
  }
}

output "instructor_index_table_name" {
  value = aws_dynamodb_table.instructor_index.name
}

