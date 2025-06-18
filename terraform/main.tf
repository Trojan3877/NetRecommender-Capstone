provider "aws" {
  region = "us-west-2"
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "netflix-recommender-cluster"
  cluster_version = "1.27"
  subnets         = ["subnet-12345678", "subnet-87654321"]
  vpc_id          = "vpc-0123456789abcdef0"
  manage_aws_auth = true
  node_groups = {
    default = {
      desired_capacity = 2
      max_capacity     = 3
      min_capacity     = 1
      instance_types   = ["t3.medium"]
    }
  }
}

resource "helm_release" "netflix" {
  name       = "netflix"
  repository = "https://charts.helm.sh/stable"
  chart      = "../helm/netflix-recommender"
  version    = "0.1.0"

  values = [
    file("../helm/netflix-recommender/values.yaml")
  ]
}
