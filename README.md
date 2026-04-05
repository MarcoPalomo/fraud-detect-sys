# fraud-detect-sys

Système de détection de fraude en temps réel sur K3s.
Pipeline complet : ingestion Kafka → embeddings PyTorch → scoring FastAPI → alerting → retraining automatique.

---

## Table des matières

1. [Stack technique](#1-stack-technique)
2. [Architecture](#2-architecture)
3. [Arborescence](#3-arborescence)
4. [Infrastructure K3s](#4-infrastructure-k3s)
5. [Deep Learning](#5-deep-learning)
6. [API de scoring](#6-api-de-scoring)
7. [Ingestion Kafka](#7-ingestion-kafka)
8. [Chaîne CI/CD](#8-chaîne-cicd)
9. [Observabilité](#9-observabilité)
10. [Déploiement pas à pas](#10-déploiement-pas-à-pas)
11. [Variables d'environnement](#11-variables-denvironnement)

---

## 1. Stack technique

| Rôle | Outil |
|---|---|
| Ingestion temps réel | Kafka via **Strimzi Operator** (KRaft, single-broker) |
| Orchestration pipelines ML | **Argo Workflows** |
| Déclenchement CD | **Argo Events** (webhook GitLab) |
| Training / Embeddings | **PyTorch + PyTorch Lightning** |
| Experiment tracking | **MLflow** (tracking + registry) |
| Stockage vectoriel | **pgvector** (PostgreSQL 16) |
| Serving | **FastAPI** + Deployment K3s |
| UI de test | **Gradio** (embarqué dans le pod serving) |
| Observabilité | **OpenTelemetry + Loki + Prometheus + Grafana** |
| CI | **GitLab CI** (lint, scan, test, build) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        K3s Cluster                          │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────┐   │
│  │  Kafka   │───▶│ PyTorch  │───▶│   FastAPI Serving   │   │
│  │ (Strimzi)│    │  Model   │    │   (score de risque) │   │
│  └──────────┘    └────┬─────┘    └──────────┬──────────┘   │
│                       │                     │               │
│              ┌────────▼──────┐    ┌─────────▼──────────┐   │
│              │    MLflow     │    │  pgvector (PG)      │   │
│              │  (tracking +  │    │  (embeddings store) │   │
│              │   registry)   │    └────────────────────┘   │
│              └───────────────┘                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Argo Workflows (pipelines ML)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   OpenTelemetry → Loki + Prometheus → Grafana        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Chaîne CI/CD complète

```
GitHub Push / PR
      │
      ▼  (webhook)
GitLab CI
  ├── lint     ruff + mypy
  ├── scan     bandit (SAST) + trivy (CVE images)
  ├── test     pytest + coverage
  └── build    docker build + push registry GitLab
                    │
                    │ POST webhook (trunk only)
                    ▼
             Argo Events
              Sensor
                    │
                    ▼
           Argo CD Pipeline
  ├── anonymize-data   HMAC IDs, floutage GPS, masquage IP/carte
  ├── train            PyTorch Lightning + MLflow
  ├── evaluate         F1/AUC-ROC → promotion Production si F1 ≥ 0.80
  └── deploy           rolling restart Deployment serving
```

---

## 3. Arborescence

```
fraud-detect-sys/
├── .gitlab-ci.yml               Pipeline CI GitLab
├── requirements.model.txt       Dépendances training
├── requirements.serving.txt     Dépendances serving
│
├── core/
│   ├── data/
│   │   ├── kafka/
│   │   │   ├── producer.py      Envoi transactions + simulateur dev
│   │   │   └── consumer.py      Lecture, scoring, routage fraud-alerts
│   │   ├── pgvector/
│   │   │   ├── client.py        Stockage et recherche nearest-neighbor
│   │   │   └── schema.sql       Table embeddings, index HNSW, vue recent_fraud
│   │   └── utils/
│   │       ├── preprocessing.py Nettoyage, SMOTE, WeightedRandomSampler
│   │       └── anonymization.py HMAC IDs, floutage geo, masquage PII
│   ├── model/
│   │   ├── model.py             FraudMLP + FraudDetector (Lightning)
│   │   ├── embeddings.py        EmbeddingGenerator + PgVectorClient
│   │   ├── train.py             Script d'entraînement (env vars → Argo)
│   │   └── eval.py              Évaluation + promotion MLflow Registry
│   ├── mlflow/
│   │   ├── tracking.py          Configuration MLflow Tracking
│   │   └── registry.py          Gestion staging/production
│   ├── serving/
│   │   ├── api.py               FastAPI : /predict, /metrics, /healthz
│   │   └── gradio_ui.py         UI de test manuelle
│   └── observability/
│       ├── otel.py              Configuration tracing distribué
│       └── metrics.py           Métriques Prometheus
│
├── k8s/
│   ├── namespace.yaml           Namespace fraud-detect
│   ├── kafka/
│   │   ├── kafka-cluster.yaml   Strimzi KRaft single-broker
│   │   ├── kafka-topic.yaml     Topics fraud-transactions + fraud-alerts
│   │   └── consumer-deployment.yaml
│   ├── pgvector/
│   │   ├── statefulset.yaml     PostgreSQL + pgvector, PVC 10Gi
│   │   ├── service.yaml         Service headless
│   │   └── secret.yaml          Credentials DB
│   ├── mlflow/
│   │   ├── deployment.yaml      MLflow server (backend pgvector)
│   │   └── service.yaml         Service + PVC 20Gi + Ingress
│   ├── serving/
│   │   ├── deployment.yaml      FastAPI + annotations Prometheus
│   │   └── ingress.yaml         Traefik : api/ui/mlflow.fraud.local
│   └── monitoring/
│       ├── loki-stack.yaml      Loki + Promtail
│       ├── prometheus.yaml      Prometheus
│       └── grafana.yaml         Grafana (sources Loki + Prometheus)
│
├── argo/
│   ├── pipelines/
│   │   ├── train-eval-deploy.yaml   WorkflowTemplate ML déclenché manuellement
│   │   └── retrain-cron.yaml        CronWorkflow hebdomadaire (lundi 02h00)
│   ├── cd/
│   │   └── cd-pipeline.yaml         WorkflowTemplate CD (déclenché par GitLab)
│   └── events/
│       ├── eventsource.yaml         Webhook GitLab → Argo Events
│       ├── sensor.yaml              Sensor → crée Workflow CD
│       └── secret.yaml              Token auth webhook
│
└── docker/
    ├── Dockerfile.model             Image training PyTorch Lightning
    └── Dockerfile.serving           Image FastAPI + Gradio
```

---

## 4. Infrastructure K3s

### Ressources minimales

| Ressource | Minimum | Recommandé |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 16 GB | 32 GB |
| Disque | 50 GB SSD | 100 GB SSD |
| GPU | optionnel | optionnel |

### Hosts locaux (à ajouter dans `/etc/hosts`)

```
<IP_K3S>  api.fraud.local
<IP_K3S>  ui.fraud.local
<IP_K3S>  mlflow.fraud.local
```

---

## 5. Deep Learning

### Modèle — `FraudDetector` (PyTorch Lightning)

Architecture MLP 4 couches :

```
Input (7) → Linear(128) + BN + ReLU + Dropout
          → Linear(64)  + BN + ReLU + Dropout
          → Linear(32)  + BN + ReLU        ← embedding 32D stocké dans pgvector
          → Linear(1)                       ← logit (BCEWithLogitsLoss)
```

**Features d'entrée (7) :** `amount`, `frequency_1h`, `frequency_24h`, `latitude`, `longitude`, `transaction_type`, `device_type`

### Gestion du déséquilibre de classes

Deux mécanismes combinés :
- **SMOTE** : suréchantillonnage de la classe fraude à 20% sur le train set
- **WeightedRandomSampler** : rééquilibrage des batchs au DataLoader
- **pos_weight** dans `BCEWithLogitsLoss` : pénalise davantage les faux négatifs

### Métriques suivies dans MLflow

`train/loss`, `val/loss`, `val/f1`, `val/auroc`, `val/precision`, `val/recall`, `test/f1`, `test/auroc`

### Promotion automatique

`eval.py` promeut le modèle en **Production** dans MLflow Registry si `test/f1 ≥ F1_PROMOTION_THRESHOLD` (défaut `0.80`). Sinon `sys.exit(1)` → Argo annule le step deploy.

---

## 6. API de scoring

**Base URL :** `http://api.fraud.local`

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/predict` | Score de risque d'une transaction |
| `GET` | `/healthz` | Liveness probe K8s |
| `GET` | `/readyz` | Readiness probe (vérifie que le modèle est chargé) |
| `POST` | `/reload-model` | Rechargement à chaud depuis MLflow Registry |
| `GET` | `/metrics` | Métriques Prometheus |

### Exemple `/predict`

```bash
curl -X POST http://api.fraud.local/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-001",
    "amount": 4500.00,
    "frequency_1h": 12,
    "frequency_24h": 45,
    "latitude": 48.8566,
    "longitude": 2.3522,
    "transaction_type": "online",
    "device_type": "mobile"
  }'
```

```json
{
  "transaction_id": "tx-001",
  "risk_score": 0.8734,
  "is_fraud": true,
  "threshold": 0.5,
  "model_version": "3"
}
```

---

## 7. Ingestion Kafka

### Topics

| Topic | Partitions | Rétention | Usage |
|---|---|---|---|
| `fraud-transactions` | 3 | 7 jours | Transactions entrantes |
| `fraud-alerts` | 1 | 30 jours | Alertes fraude (score ≥ 0.7) |

### Simulateur de transactions (dev)

```bash
python -m core.data.kafka.producer --rate 10   # 10 transactions/seconde
```

### Bootstrap servers

```
fraud-kafka-kafka-bootstrap.fraud-detect.svc:9092
```

---

## 8. Chaîne CI/CD

### GitLab CI

| Stage | Job | Outil | Branches |
|---|---|---|---|
| lint | `ruff` | Style + format | MR + trunk |
| lint | `mypy` | Type checking strict | MR + trunk |
| scan | `bandit` | SAST Python HIGH/CRITICAL | MR + trunk |
| scan | `trivy-*` | CVE images Docker | trunk |
| test | `pytest` | Coverage + JUnit | MR + trunk |
| build | `build-model` | docker build + push | trunk (si fichiers changés) |
| build | `build-serving` | docker build + push | trunk (si fichiers changés) |
| trigger | `trigger-argo-cd` | Webhook → Argo Events | trunk (après build + test) |

### CD Pipeline Argo (`fraud-cd-pipeline`)

```
anonymize-data → train → evaluate → deploy
                                  ↑
                         (conditionnel F1 ≥ 0.80)
```

### Retraining automatique

`CronWorkflow` tous les **lundis à 02h00** (Europe/Paris) :
```
export pgvector→CSV → train → evaluate → deploy
```

### Variables GitLab CI à configurer

| Variable | Description |
|---|---|
| `ARGO_EVENTS_WEBHOOK_URL` | `http://<IP_K3S>:12000` |
| `ARGO_EVENTS_TOKEN` | Token partagé avec `argo/events/secret.yaml` |
| `CI_REGISTRY` | Registry GitLab (auto-injecté) |

---

## 9. Observabilité

| Signal | Outil | Accès |
|---|---|---|
| Traces | OpenTelemetry → Tempo | Grafana |
| Logs | Loki + Promtail | Grafana |
| Métriques | Prometheus | Grafana / `api.fraud.local/metrics` |

### Métriques exposées par l'API

| Métrique | Type | Description |
|---|---|---|
| `fraud_requests_total` | Counter | Nb requêtes par résultat (fraud/legit) |
| `fraud_request_latency_seconds` | Histogram | Latence de scoring |
| `fraud_score_distribution` | Histogram | Distribution des scores de risque |

---

## 10. Déploiement pas à pas

### Prérequis

```bash
# K3s
curl -sfL https://get.k3s.io | sh -

# Argo Workflows
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

# Argo Events
kubectl create namespace argo-events
kubectl apply -f https://github.com/argoproj/argo-events/releases/latest/download/install.yaml

# Strimzi Operator
kubectl create namespace fraud-detect
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi-operator strimzi/strimzi-kafka-operator -n fraud-detect
```

### Appliquer les manifests

```bash
# Namespace + secrets
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pgvector/secret.yaml
kubectl apply -f argo/events/secret.yaml
kubectl apply -f argo/cd/cd-pipeline.yaml   # contient le secret anonymization

# Infrastructure
kubectl apply -f k8s/pgvector/
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/kafka/

# Schema pgvector (une seule fois)
kubectl exec -n fraud-detect -it pgvector-0 -- \
  psql -U fraud_user -d fraud -f /docker-entrypoint-initdb.d/schema.sql

# Serving
kubectl apply -f k8s/serving/

# Monitoring
kubectl apply -f k8s/monitoring/

# Argo pipelines + events
kubectl apply -f argo/pipelines/
kubectl apply -f argo/cd/
kubectl apply -f argo/events/
```

### Déclencher un entraînement manuellement

```bash
argo submit --from workflowtemplate/fraud-train-eval-deploy \
  -p data-path=/data/transactions.csv \
  -n fraud-detect \
  --watch
```

---

## 11. Variables d'environnement

### Serving (`fraud-serving`)

| Variable | Défaut | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URI MLflow |
| `PGVECTOR_HOST` | `pgvector` | Hôte PostgreSQL |
| `PGVECTOR_DB` | `fraud` | Base de données |
| `FRAUD_THRESHOLD` | `0.5` | Seuil de classification fraude |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | Endpoint OTel |

### Consumer Kafka

| Variable | Défaut | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Brokers Kafka |
| `SERVING_API_URL` | `http://fraud-serving:8000` | URL API scoring |
| `ALERT_THRESHOLD` | `0.7` | Seuil d'alerte vers fraud-alerts |

### Training (Argo Job)

| Variable | Défaut | Description |
|---|---|---|
| `DATA_PATH` | — | Chemin CSV (obligatoire) |
| `MAX_EPOCHS` | `50` | Nombre d'epochs |
| `POS_WEIGHT` | `10.0` | Poids classe fraude |
| `F1_PROMOTION_THRESHOLD` | `0.80` | Seuil promotion Production |
| `ANON_SECRET_KEY` | — | Clé HMAC anonymisation (obligatoire) |
