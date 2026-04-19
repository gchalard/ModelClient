// Build from the ModelRunner directory:
//   docker buildx bake -f docker-bake.hcl
//
// Context is this directory (ModelRunner).

variable "TAG" {
  default = "local"
}

group "default" {
  targets = ["modelrunner"]
}

target "modelrunner" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["modelrunner:${TAG}"]
  labels = {
    "org.opencontainers.image.title" = "modelrunner"
  }
}
