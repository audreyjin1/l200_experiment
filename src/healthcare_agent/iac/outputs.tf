output "cloud_run_url" {
  description = "The invocation URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.healthcare_agent.uri
}

output "dlp_deidentify_template_id" {
  description = "The Resource ID of the Cloud DLP surrogate de-identification template"
  value       = google_data_loss_prevention_deidentify_template.phi_deidentify_template.id
}

output "firestore_database_name" {
  description = "The Firestore database name for session persistence"
  value       = google_firestore_database.session_db.name
}
