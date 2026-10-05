/**
 * Terraform Infrastructure as Code (IaC) for Healthcare Agent ADK 2.0 on Google Cloud.
 * Deploys Cloud Run, Cloud DLP de-identify template, Firestore, and IAM roles.
 */

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required GCP APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "dlp.googleapis.com",
    "aiplatform.googleapis.com",
    "firestore.googleapis.com",
    "secretmanager.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Service Account for Healthcare Agent Cloud Run Service
resource "google_service_account" "agent_sa" {
  account_id   = "healthcare-agent-sa"
  display_name = "Healthcare ADK Agent Runtime Service Account"
  project      = var.project_id
}

# 3. IAM Bindings for DLP, Vertex AI, and Firestore
resource "google_project_iam_member" "dlp_user" {
  project = var.project_id
  role    = "roles/dlp.user"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "datastore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

# 4. Cloud DLP De-Identify Template for PHI Surrogate Tokenization
resource "google_data_loss_prevention_deidentify_template" "phi_deidentify_template" {
  parent       = "projects/${var.project_id}/locations/${var.region}"
  display_name = "Healthcare PHI Surrogate De-identification Template"
  description  = "Tokenizes patient names, MRNs, phone numbers, and dates into surrogate placeholders"

  deidentify_config {
    info_type_transformations {
      transformations {
        info_types {
          name = "PERSON_NAME"
        }
        info_types {
          name = "PHONE_NUMBER"
        }
        info_types {
          name = "EMAIL_ADDRESS"
        }
        info_types {
          name = "US_SOCIAL_SECURITY_NUMBER"
        }
        info_types {
          name = "DATE_OF_BIRTH"
        }
        primitive_transformation {
          replace_with_info_type_config = true
        }
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

# 5. Firestore Database for Patient Session State
resource "google_firestore_database" "session_db" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  concurrency_mode = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"

  depends_on = [google_project_service.required_apis]
}

# 6. Cloud Run Service for Healthcare Agent API
resource "google_cloud_run_v2_service" "healthcare_agent" {
  name     = "healthcare-agent-adk"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.agent_sa.email

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "STORAGE_BACKEND"
        value = "firestore"
      }
      env {
        name  = "GEMINI_MODEL"
        value = "gemini-2.5-flash"
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_firestore_database.session_db,
  ]
}
