# =============================================================================
# fraud-detect-sys — Makefile
#
# Usage :
#   make help          Liste toutes les cibles
#   make bootstrap     Installation complète depuis zéro
#   make up            Applique toute la stack K8s
#   make down          Supprime toutes les ressources K8s
#   make build         Build + import des images Docker dans K3s
#   make train         Déclenche un run d'entraînement via Argo
#   make test          Lance les tests unitaires
#   make lint          Lint + type checking
# =============================================================================

NAMESPACE       := fraud-detect
KUBECONFIG      ?= /etc/rancher/k3s/k3s.yaml
IMAGE_MODEL     := fraud-detect/model:latest
IMAGE_SERVING   := fraud-detect/serving:latest
DATA_PATH       ?= /data/transactions.csv

export KUBECONFIG

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "  fraud-detect-sys"
	@echo ""
	@echo "  Bootstrap"
	@echo "    make bootstrap          Installation complète (K3s + operators + stack)"
	@echo "    make install-operators  Installe Argo Workflows/Events + Strimzi"
	@echo "    make init-db            Initialise le schema pgvector"
	@echo ""
	@echo "  Stack K8s"
	@echo "    make up                 Applique toute la stack"
	@echo "    make down               Supprime toutes les ressources du namespace"
	@echo "    make status             Etat des pods"
	@echo "    make restart            Rolling restart de fraud-serving"
	@echo ""
	@echo "  Images Docker"
	@echo "    make build              Build model + serving et import dans K3s"
	@echo "    make build-model        Build uniquement l'image model"
	@echo "    make build-serving      Build uniquement l'image serving"
	@echo ""
	@echo "  ML"
	@echo "    make train              Déclenche un run Argo train→eval→deploy"
	@echo "    make simulate           Lance le simulateur Kafka (10 tx/s)"
	@echo ""
	@echo "  Dev"
	@echo "    make test               Lance pytest"
	@echo "    make lint               ruff + mypy"
	@echo "    make fmt                Formate le code avec ruff"
	@echo "    make install-dev        Installe les dépendances dev"
	@echo ""
	@echo "  Secrets"
	@echo "    make secrets            Applique les secrets K8s (remplir .env d'abord)"
	@echo ""

# -----------------------------------------------------------------------------
# Bootstrap complet
# -----------------------------------------------------------------------------
.PHONY: bootstrap
bootstrap: install-k3s install-operators build up wait-ready init-db
	@echo ""
	@echo "  Stack démarrée."
	@echo "  Ajoute dans /etc/hosts : $(shell hostname -I | awk '{print $$1}') api.fraud.local ui.fraud.local mlflow.fraud.local grafana.fraud.local"
	@echo ""

.PHONY: install-k3s
install-k3s:
	@if ! command -v k3s > /dev/null; then \
		echo "→ Installation K3s..."; \
		curl -sfL https://get.k3s.io | sh -; \
	else \
		echo "→ K3s déjà installé, skip."; \
	fi

.PHONY: install-operators
install-operators:
	@echo "→ Argo Workflows..."
	kubectl create namespace argo --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -n argo \
	  -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

	@echo "→ Argo Events..."
	kubectl create namespace argo-events --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f \
	  https://github.com/argoproj/argo-events/releases/latest/download/install.yaml
	kubectl apply -n argo-events -f \
	  https://github.com/argoproj/argo-events/releases/latest/download/eventbus-native.yaml

	@echo "→ Strimzi Operator..."
	kubectl apply -f \
	  "https://strimzi.io/install/latest?namespace=$(NAMESPACE)"
	@echo "→ Operators installés."

# -----------------------------------------------------------------------------
# Stack K8s
# -----------------------------------------------------------------------------
.PHONY: up
up: secrets
	@echo "→ Namespace..."
	kubectl apply -f k8s/namespace.yaml

	@echo "→ pgvector..."
	kubectl apply -f k8s/pgvector/

	@echo "→ MLflow..."
	kubectl apply -f k8s/mlflow/

	@echo "→ Kafka..."
	kubectl apply -f k8s/kafka/

	@echo "→ Monitoring..."
	kubectl apply -f k8s/monitoring/

	@echo "→ Serving..."
	kubectl apply -f k8s/serving/

	@echo "→ Argo pipelines + events..."
	kubectl apply -f argo/pipelines/
	kubectl apply -f argo/cd/
	kubectl apply -f argo/events/

	@echo "→ Stack appliquée."

.PHONY: down
down:
	@echo "→ Suppression du namespace $(NAMESPACE)..."
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	@echo "→ Suppression des CRDs Kafka..."
	kubectl delete -f k8s/kafka/kafka-cluster.yaml --ignore-not-found

.PHONY: status
status:
	kubectl get pods -n $(NAMESPACE) -o wide

.PHONY: restart
restart:
	kubectl rollout restart deployment/fraud-serving -n $(NAMESPACE)
	kubectl rollout status  deployment/fraud-serving -n $(NAMESPACE)

.PHONY: wait-ready
wait-ready:
	@echo "→ Attente pgvector..."
	kubectl wait --for=condition=ready pod -l app=pgvector \
	  -n $(NAMESPACE) --timeout=180s
	@echo "→ Attente MLflow..."
	kubectl wait --for=condition=ready pod -l app=mlflow \
	  -n $(NAMESPACE) --timeout=180s
	@echo "→ Tous les pods essentiels sont Ready."

# -----------------------------------------------------------------------------
# Secrets (lit depuis .env)
# -----------------------------------------------------------------------------
.PHONY: secrets
secrets:
	@test -f .env || (echo "ERREUR : fichier .env manquant. Copie .env.example et remplis-le." && exit 1)
	@echo "→ Application des secrets..."
	@set -a && . ./.env && set +a && \
	  kubectl create secret generic pgvector-secret \
	    --from-literal=username=$$PGVECTOR_USER \
	    --from-literal=password=$$PGVECTOR_PASSWORD \
	    --namespace=$(NAMESPACE) \
	    --dry-run=client -o yaml | kubectl apply -f - && \
	  kubectl create secret generic argo-events-token \
	    --from-literal=token=$$ARGO_EVENTS_TOKEN \
	    --namespace=$(NAMESPACE) \
	    --dry-run=client -o yaml | kubectl apply -f - && \
	  kubectl create secret generic anonymization-secret \
	    --from-literal=secret-key=$$ANON_SECRET_KEY \
	    --namespace=$(NAMESPACE) \
	    --dry-run=client -o yaml | kubectl apply -f - && \
	  kubectl create secret generic grafana-secret \
	    --from-literal=admin-password=$$GRAFANA_PASSWORD \
	    --namespace=$(NAMESPACE) \
	    --dry-run=client -o yaml | kubectl apply -f -
	@echo "→ Secrets appliqués."

# -----------------------------------------------------------------------------
# Images Docker
# -----------------------------------------------------------------------------
.PHONY: build
build: build-model build-serving

.PHONY: build-model
build-model:
	@echo "→ Build image model..."
	docker build -f docker/Dockerfile.model -t $(IMAGE_MODEL) .
	@echo "→ Import dans K3s..."
	docker save $(IMAGE_MODEL) | sudo k3s ctr images import -

.PHONY: build-serving
build-serving:
	@echo "→ Build image serving..."
	docker build -f docker/Dockerfile.serving -t $(IMAGE_SERVING) .
	@echo "→ Import dans K3s..."
	docker save $(IMAGE_SERVING) | sudo k3s ctr images import -

# -----------------------------------------------------------------------------
# Base de données
# -----------------------------------------------------------------------------
.PHONY: init-db
init-db:
	@echo "→ Initialisation du schema pgvector..."
	kubectl exec -n $(NAMESPACE) pgvector-0 -- \
	  psql -U fraud_user -d fraud \
	  -c "$$(cat core/data/pgvector/schema.sql)"
	@echo "→ Schema initialisé."

# -----------------------------------------------------------------------------
# ML
# -----------------------------------------------------------------------------
.PHONY: train
train:
	@echo "→ Soumission du workflow Argo train→eval→deploy..."
	argo submit --from workflowtemplate/fraud-train-eval-deploy \
	  -p data-path=$(DATA_PATH) \
	  -n $(NAMESPACE) \
	  --watch

.PHONY: simulate
simulate:
	@echo "→ Simulateur Kafka (10 tx/s) — Ctrl+C pour arrêter..."
	kubectl run kafka-sim --rm -it \
	  --image=$(IMAGE_SERVING) \
	  --namespace=$(NAMESPACE) \
	  --env="KAFKA_BOOTSTRAP_SERVERS=fraud-kafka-kafka-bootstrap:9092" \
	  --restart=Never \
	  -- python -m core.data.kafka.producer --rate 10

# -----------------------------------------------------------------------------
# Dev
# -----------------------------------------------------------------------------
.PHONY: install-dev
install-dev:
	pip install -r requirements.serving.txt -r requirements.model.txt
	pip install pytest pytest-cov pytest-asyncio httpx ruff mypy

.PHONY: test
test:
	pytest tests/ --cov=core --cov-report=term-missing -v

.PHONY: lint
lint:
	ruff check core/
	mypy core/ --ignore-missing-imports

.PHONY: fmt
fmt:
	ruff format core/
	ruff check core/ --fix

# -----------------------------------------------------------------------------
# Logs / Debug
# -----------------------------------------------------------------------------
.PHONY: logs-serving
logs-serving:
	kubectl logs -n $(NAMESPACE) -l app=fraud-serving --tail=100 -f

.PHONY: logs-consumer
logs-consumer:
	kubectl logs -n $(NAMESPACE) -l app=fraud-consumer --tail=100 -f

.PHONY: logs-kafka
logs-kafka:
	kubectl logs -n $(NAMESPACE) -l app=fraud-kafka --tail=50 -f
