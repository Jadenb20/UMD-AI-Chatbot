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
        Sid      = "InvokeBedrock"
        Effect   = "Allow"
        Action   = ["bedrock:GetModel", "bedrock:InvokeModel"]
        Resource = "*"
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
        Sid    = "AccessOpenSearch"
        Effect = "Allow"
        Action = [
          "aoss:APIAccessAll"
        ]
        Resource = "*"
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
  source_file = "${path.module}/../backend/lambda/chat.py"
  output_path = "${path.module}/../backend/lambda/chat.zip"
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