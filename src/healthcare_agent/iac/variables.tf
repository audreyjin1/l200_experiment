variable "project_id" {
  description = "The Google Cloud Project ID"
  type        = string
  default     = "healthcare-adk-prod"
}

variable "region" {
  description = "The GCP region for deployments"
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "The Artifact Registry container image URI for the Healthcare Agent"
  type        = string
  default     = "us-central1-docker.pkg.dev/healthcare-adk-prod/healthcare-agent/agent:latest"
}
