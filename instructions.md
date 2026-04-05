# instructions.md

## 1. Objectif

- **Détection d'Anomalies** :
    - Identifier des schémas inhabituels (fraude, erreurs système).
    - Données : logs, transactions.

---

## 2. Stack Technique (K3s-optimisée)

| Rôle | Outil |
|---|---|
| Ingestion temps réel | **Kafka** via Strimzi Operator (KRaft, single-broker) |
| Orchestration pipelines ML | **Argo Workflows** |
| Training / Embeddings | **PyTorch + PyTorch Lightning** (Argo Job) |
| Experiment tracking | **MLflow** (tracking + registry) |
| Stockage vectoriel | **pgvector** (PostgreSQL) + FAISS en lib dans le pod |
| Serving | **FastAPI** + Deployment K3s |
| UI de test | **Gradio** (embarqué dans le pod serving) |
| Observabilité | **OpenTelemetry Collector + Loki + Prometheus + Grafana** |

---

## 3. Arborescence du Repository

```
fraud-detect-sys/
├── README.md
├── docs/
├── core/
│   ├── data/
│   │   ├── kafka/              (Strimzi CRDs + producer/consumer)
│   │   ├── pgvector/           (schema SQL + client)
│   │   └── utils/
│   ├── model/
│   │   ├── model.py            (PyTorch Lightning module)
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── embeddings.py
│   ├── mlflow/
│   │   ├── tracking.py
│   │   └── registry.py
│   ├── serving/
│   │   ├── api.py              (FastAPI)
│   │   └── gradio_ui.py        (UI de test)
│   └── observability/
│       ├── otel.py
│       └── metrics.py
├── k8s/
│   ├── kafka/                  (Strimzi CRs)
│   ├── mlflow/                 (Deployment + Service)
│   ├── pgvector/               (StatefulSet)
│   ├── serving/                (Deployment + Ingress)
│   └── monitoring/             (Loki + Grafana + Prometheus)
├── argo/
│   └── pipelines/              (WorkflowTemplates : train, eval, deploy)
├── docker/
│   ├── Dockerfile.model
│   └── Dockerfile.serving
└── tests/
```

---

## 4. Architecture

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
│              │    MLflow     │    │  pgvector (PG)     │   │
│              │  (tracking +  │    │  (embeddings store)│   │
│              │   registry)   │    └────────────────────┘   │
│              └───────────────┘                             │
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

---

## 5. Cas d'Usage : Détection de Fraude

- **Données** : transactions (montant, fréquence, géolocalisation, type, device info, etc.)
- **Objectif** : identifier des transactions potentiellement frauduleuses en temps réel ou post-traitement.
- **Défis** : rareté des fraudes, similarité avec transactions légitimes.

### Stratégies de Modélisation

- **Supervisé** : Logistic Regression, Random Forest, Gradient Boosting.
- **Non Supervisé** : Isolation Forest, One-Class SVM.
- **Hybride** : combinaison des deux (recommandé).

### Features Importantes

- Montant, fréquence, géolocalisation, type, device info.

### Métriques Clés (MLflow)

- Précision, rappel, F1-score, AUC-ROC.

---

## 6. Workflow

1. **Ingestion** — Kafka (Strimzi) consomme les transactions en temps réel
2. **Préparation** — nettoyage, normalisation, équilibrage (SMOTE)
3. **Embedding** — PyTorch Lightning génère les embeddings, stockés dans pgvector
4. **Modélisation** — entraînement déclenché via Argo Workflow
5. **Évaluation** — métriques loguées dans MLflow, déploiement conditionnel sur F1-score
6. **Serving** — FastAPI expose le score de risque, UI Gradio pour tests
7. **Alerting** — alerte si score élevé
8. **Observabilité** — OTel → Loki + Prometheus → Grafana

---

## 7. Infrastructure K3s

### Ressources minimales

```
CPU  : 4 cores (8 recommandé)
RAM  : 16 GB (32 recommandé)
Disk : 50 GB SSD
GPU  : optionnel (CPU training suffisant pour les modèles proposés)
```

### Exemple pipeline Argo (train → eval → deploy conditionnel)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
    - name: train
    - name: evaluate
    - name: deploy-if-better   # conditionnel sur F1-score MLflow
```

---

## 8. Points d'Attention

- Équilibrage des données (fraudes rares) — SMOTE, pondération, ou few-shot learning
- Adaptation continue (fraudeurs évoluent) — retraining périodique via Argo CronWorkflow
- Collaboration avec équipes sécurité
