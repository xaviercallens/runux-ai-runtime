# ═══════════════════════════════════════════════════════════════════
# SymBrain v4 — Terraform Infrastructure for GCP
# ═══════════════════════════════════════════════════════════════════
#
# Provisions:
#   1. Artifact Registry repository for Docker images
#   2. Cloud Run service (Edge-7B / Cloud-32B with L4 GPU)
#   3. GCE instance for 70B/122B with A100 GPUs (on-demand)
#   4. IAM service accounts and permissions
#   5. Cloud Build triggers for CI/CD
#
# Usage:
#   cd deploy/terraform
#   terraform init
#   terraform plan -var="project_id=gen-lang-client-0625573011"
#   terraform apply -var="project_id=gen-lang-client-0625573011"
# ═══════════════════════════════════════════════════════════════════

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.80.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.80.0"
    }
  }
}

# ── Variables ──────────────────────────────────────────────────────

variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "gen-lang-client-0625573011"
}

variable "region" {
  description = "Primary region for Cloud Run"
  type        = string
  default     = "europe-west1"
}

variable "region_gpu" {
  description = "Region with GPU availability for GCE instances"
  type        = string
  default     = "us-central1"
}

variable "zone_gpu" {
  description = "Zone with GPU availability"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment label (dev, staging, prod)"
  type        = string
  default     = "prod"
}

# ── Provider ──────────────────────────────────────────────────────

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Enable Required APIs ─────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
  ])
  project = var.project_id
  service = each.value
  disable_on_destroy = false
}

# ═══════════════════════════════════════════════════════════════════
#  1. Artifact Registry
# ═══════════════════════════════════════════════════════════════════

resource "google_artifact_registry_repository" "symbrain" {
  provider      = google-beta
  location      = var.region
  repository_id = "symbrain-v4"
  format        = "DOCKER"
  description   = "SymBrain v4 inference server container images"

  labels = {
    environment = var.environment
    component   = "symbrain-v4"
  }

  depends_on = [google_project_service.apis]
}

# ═══════════════════════════════════════════════════════════════════
#  2. Service Account for Cloud Run
# ═══════════════════════════════════════════════════════════════════

resource "google_service_account" "symbrain_runner" {
  account_id   = "symbrain-v4-runner"
  display_name = "SymBrain v4 Cloud Run Service Account"
  description  = "Service account for SymBrain v4 inference server"
}

resource "google_project_iam_member" "symbrain_runner_roles" {
  for_each = toset([
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.symbrain_runner.email}"
}

# ═══════════════════════════════════════════════════════════════════
#  3. Cloud Run Service — Edge-7B (CPU-only, simulation mode)
# ═══════════════════════════════════════════════════════════════════

resource "google_cloud_run_v2_service" "symbrain_edge" {
  provider = google-beta
  name     = "symbrain-v4-edge"
  location = var.region

  template {
    service_account = google_service_account.symbrain_runner.email

    scaling {
      min_instance_count = 0
      max_instance_count = 4
    }

    containers {
      name  = "symbrain-v4"
      image = "${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "MODEL_TIER"
        value = "7B"
      }
      env {
        name  = "SIMULATION_MODE"
        value = "true"
      }
      env {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/v4/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/v4/health"
          port = 8080
        }
        period_seconds = 30
      }
    }

    timeout = "300s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = {
    environment = var.environment
    tier        = "edge-7b"
    component   = "symbrain-v4"
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.symbrain,
  ]
}

# ── Allow unauthenticated access (for demo / testing) ────────────

resource "google_cloud_run_v2_service_iam_member" "edge_public" {
  provider = google-beta
  name     = google_cloud_run_v2_service.symbrain_edge.name
  location = google_cloud_run_v2_service.symbrain_edge.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ═══════════════════════════════════════════════════════════════════
#  4. Cloud Run Service — Cloud-32B (L4 GPU)
# ═══════════════════════════════════════════════════════════════════

resource "google_cloud_run_v2_service" "symbrain_cloud32" {
  provider = google-beta
  name     = "symbrain-v4-cloud32"
  location = var.region

  template {
    service_account = google_service_account.symbrain_runner.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    node_selector {
      accelerator = "nvidia-l4"
    }

    containers {
      name  = "symbrain-v4"
      image = "${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "MODEL_TIER"
        value = "32B"
      }
      env {
        name  = "SIMULATION_MODE"
        value = "false"
      }
      env {
        name  = "HF_HOME"
        value = "/app/model_cache"
      }
      env {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      }

      resources {
        limits = {
          cpu    = "8"
          memory = "32Gi"
          "nvidia.com/gpu" = "1"
        }
      }

      startup_probe {
        http_get {
          path = "/v4/health"
          port = 8080
        }
        initial_delay_seconds = 60
        period_seconds        = 15
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/v4/health"
          port = 8080
        }
        period_seconds    = 60
        timeout_seconds   = 10
        failure_threshold = 3
      }
    }

    timeout = "900s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = {
    environment = var.environment
    tier        = "cloud-32b"
    component   = "symbrain-v4"
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.symbrain,
  ]
}

# ═══════════════════════════════════════════════════════════════════
#  5. GCE Instance — Cloud-70B (2×A100-80GB, on-demand)
# ═══════════════════════════════════════════════════════════════════

resource "google_compute_instance" "symbrain_70b" {
  provider     = google-beta
  name         = "symbrain-v4-70b"
  machine_type = "a2-ultragpu-2g"
  zone         = var.zone_gpu

  # Start stopped — user activates on demand
  desired_status = "TERMINATED"

  boot_disk {
    initialize_params {
      image = "projects/ml-images/global/images/c0-deeplearning-common-gpu-v20240128-debian-11-py310"
      size  = 200
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = "nvidia-a100-80gb"
    count = 2
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible         = true  # Use spot pricing ($50 budget)
  }

  network_interface {
    network = "default"
    access_config {}  # External IP for SSH
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -ex
    # Install Docker and pull v4 image
    apt-get update && apt-get install -y docker.io nvidia-container-toolkit
    systemctl start docker
    # Pull and run inference server
    docker pull ${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest
    docker run -d --gpus all \
      -p 8080:8080 \
      -e MODEL_TIER=70B \
      -e SIMULATION_MODE=false \
      --name symbrain-v4-70b \
      ${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest
  EOT

  service_account {
    email  = google_service_account.symbrain_runner.email
    scopes = ["cloud-platform"]
  }

  labels = {
    environment = var.environment
    tier        = "cloud-70b"
    component   = "symbrain-v4"
  }

  tags = ["symbrain-v4", "gpu-inference"]

  depends_on = [google_project_service.apis]
}

# ── Firewall for 70B instance ────────────────────────────────────

resource "google_compute_firewall" "symbrain_inference" {
  name    = "symbrain-v4-inference-allow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080", "22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["symbrain-v4"]

  description = "Allow HTTP inference and SSH to SymBrain v4 GPU instances"
}

# ═══════════════════════════════════════════════════════════════════
#  6. Cloud Build Trigger (CI/CD)
# ═══════════════════════════════════════════════════════════════════

resource "google_cloudbuild_trigger" "symbrain_deploy" {
  provider = google-beta
  name     = "symbrain-v4-deploy"
  location = var.region

  # Manual trigger (no git repo connected)

  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "--file=deploy/Dockerfile.v4",
        "--tag=${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest",
        ".",
      ]
    }

    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "push",
        "${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest",
      ]
    }

    images = [
      "${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4/inference:latest",
    ]

    options {
      machine_type = "E2_HIGHCPU_8"
    }
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.symbrain,
  ]
}

# ═══════════════════════════════════════════════════════════════════
#  Outputs
# ═══════════════════════════════════════════════════════════════════

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/symbrain-v4"
}

output "edge_service_url" {
  value = google_cloud_run_v2_service.symbrain_edge.uri
}

output "cloud32_service_url" {
  value = google_cloud_run_v2_service.symbrain_cloud32.uri
}

output "gpu_70b_instance" {
  value = google_compute_instance.symbrain_70b.name
}

output "gpu_70b_ip" {
  value = google_compute_instance.symbrain_70b.network_interface[0].access_config[0].nat_ip
}
