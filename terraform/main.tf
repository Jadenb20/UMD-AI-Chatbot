terraform {
 required_providers {
  aws = {
   source="hashicorp/aws"
   version = "~> 5.0"
  }
 }
}

provider "aws" {
 region = "us-east-1"
}

resource "aws_s3_bucket" "frontend" {
 bucket= "umd-chatbot-frontend-${random_id.suffix.hex}"
}


resource "aws_s3_bucket" "knowledge_base" {
 bucket= "umd-chatbot-kb-${random_id.suffix.hex}"
}

resource "random_id" "suffix"{
 byte_length = 4
}