# DataNanite — GKE Deployment Summary
## Context Document for Continuing Claude Code Sessions

---

## What This Project Is

**DataNanite** (repo: `krishnarayapudi2571/Datananite`, originally cloned from `dabu810/metadataextractor`) is an AI-native metadata intelligence platform with 10 Python microservices. It uses FastAPI + LangGraph + Anthropic Claude API.

---

## Current Deployment State

| Item | Value |
|------|-------|
| GCP Project | `cog01k24f1ea555zdv7ynzthxanz5` |
| GKE Cluster | `lvp-fastapi-cluster` |
| Zone | `asia-south1-a` |
| Namespace | `datananite` |
| Node Pool | `datananite-pool` (e2-standard-2 × 2 nodes) |
| Static IP | `34.128.164.174` (reserved as `datananite-static-ip`) |
| Public URL | `http://datananite.endpoints.cog01k24f1ea555zdv7ynzthxanz5.cloud.goog` |
| GitHub Repo | `https://github.com/krishnarayapudi2571/Datananite.git` |
| Active Branch | `feat/gke-deployment` (NOT yet merged to main) |

---

## What's Done ✅

| Component | Status | Notes |
|-----------|--------|-------|
| GKE node pool created | ✅ Done | `datananite-pool`, e2-standard-2, 2 nodes |
| Namespace created | ✅ Done | `datananite` |
| All 10 K8s deployments | ✅ Done | All pods `1/1 Running` |
| All 10 K8s services | ✅ Done | ClusterIP, correct ports |
| ConfigMap | ✅ Done | `datananite-config` |
| Secrets | ✅ Done | `datananite-secrets` (has placeholder ANTHROPIC_API_KEY) |
| PVCs (9 total) | ✅ Done | Individual PVC per service, ReadWriteOnce |
| Ingress | ✅ Done | GCE ingress, static IP, host routing |
| NetworkPolicy | ✅ Done | Public UIs open, internal APIs locked |
| Static IP reserved | ✅ Done | `34.128.164.174` |
| Cloud Endpoints DNS | ✅ Done | Registered `datananite.endpoints...cloud.goog` |
| GitHub Actions workflow | ✅ Done | `.github/workflows/deploy.yml` |
| Code pushed to branch | ✅ Done | `feat/gke-deployment` |

---

## What's Still Pending ❌

| Item | What to Do |
|------|------------|
| **GitHub Secrets** | Add `GCP_SA_KEY` and `ANTHROPIC_API_KEY` to repo secrets |
| **Merge PR** | Merge `feat/gke-deployment` → `main` to trigger CI/CD |
| **CI/CD first run** | Builds all 10 Docker images, pushes to GCR, deploys |
| **Real ANTHROPIC_API_KEY** | Currently placeholder in K8s secret — needs real key |

### Add GitHub Secrets Here:
`https://github.com/krishnarayapudi2571/Datananite/settings/secrets/actions`

| Secret Name | Value |
|-------------|-------|
| `GCP_SA_KEY` | JSON contents of GCP service account key (same SA as VeriForge: `cio-ociolegalc@cog01k24f1ea555zdv7ynzthxanz5.iam.gserviceaccount.com`) |
| `ANTHROPIC_API_KEY` | Real Anthropic/Claude API key |

---

## Architecture

```
Internet
  ↓
http://datananite.endpoints.cog01k24f1ea555zdv7ynzthxanz5.cloud.goog
  ↓ (Cloud Endpoints DNS → 34.128.164.174)
GCP Global Load Balancer (static IP: datananite-static-ip)
  ↓
GKE Ingress (datananite-ingress)
  ├── /*              → chat-ui-service:8005      (main UI / orchestrator)
  ├── /tech/*         → tech-ui-service:8006      (engineer workbench)
  ├── /streamlit/*    → streamlit-ui-service:8501 (Streamlit metadata UI)
  ├── /api/agent/*    → agent-api-service:8000
  ├── /api/ontology/* → ontology-api-service:8001
  ├── /api/kg/*       → kg-api-service:8002
  ├── /api/dialog/*   → dialog-api-service:8003
  ├── /api/conformity/*→ conformity-api-service:8004
  ├── /api/shacl/*    → shacl-api-service:8007
  └── /api/unstructured/* → unstructured-api-service:8008
```

---

## All 10 Services

| Service | Port | Docker Image | Dockerfile | Purpose |
|---------|------|-------------|-----------|---------|
| agent-api | 8000 | `datananite-agent` | Dockerfile.agent | Metadata extraction from databases |
| ontology-api | 8001 | `datananite-ontology` | Dockerfile.ontology | OWL/RDF ontology generation |
| kg-api | 8002 | `datananite-kg` | Dockerfile.kg | Knowledge graph (Neo4j/Gremlin) |
| dialog-api | 8003 | `datananite-dialog` | Dockerfile.dialog | NL→SQL, GraphRAG |
| conformity-api | 8004 | `datananite-conformity` | Dockerfile.conformity | Data quality validation |
| chat-ui | 8005 | `datananite-chat` | Dockerfile.chat | Main orchestrator UI |
| tech-ui | 8006 | `datananite-tech` | Dockerfile.tech | Engineer workbench |
| shacl-api | 8007 | `datananite-shacl` | Dockerfile.shacl | SHACL validation |
| unstructured-api | 8008 | `datananite-unstructured` | Dockerfile.unstructured | Document indexing |
| streamlit-ui | 8501 | `datananite-ui` | Dockerfile.ui | Streamlit metadata UI |

All images stored at: `gcr.io/cog01k24f1ea555zdv7ynzthxanz5/datananite-{name}:latest`

---

## Key Files

```
metadataextractor/
├── .github/workflows/deploy.yml          ← CI/CD pipeline
├── deployment/
│   ├── DEPLOYMENT_SUMMARY.md             ← This file
│   ├── endpoints/
│   │   └── dns-spec.yaml                 ← Cloud Endpoints DNS config
│   └── k8s/
│       ├── namespace.yaml                ← datananite namespace
│       ├── configmap.yaml                ← Non-sensitive env vars
│       ├── secrets.yaml                  ← ANTHROPIC_API_KEY template
│       ├── pvc.yaml                      ← 9 PVCs (one per service)
│       ├── ingress.yaml                  ← GCE ingress + routing
│       ├── networkpolicy.yaml            ← Network security rules
│       ├── agent-api.yaml                ← Deployment + Service
│       ├── ontology-api.yaml             ← Deployment + Service
│       ├── kg-api.yaml                   ← Deployment + Service
│       ├── dialog-api.yaml               ← Deployment + Service
│       ├── conformity-api.yaml           ← Deployment + Service
│       ├── chat-ui.yaml                  ← Deployment + Service
│       ├── tech-ui.yaml                  ← Deployment + Service
│       ├── shacl-api.yaml                ← Deployment + Service
│       ├── unstructured-api.yaml         ← Deployment + Service
│       └── streamlit-ui.yaml             ← Deployment + Service
```

---

## K8s Deployment Pattern (All Services Follow This)

```yaml
strategy:
  type: Recreate                    # ← IMPORTANT: must stay Recreate (not RollingUpdate)
                                    # RollingUpdate causes PVC Multi-Attach errors
initContainers:
- name: fix-permissions
  image: busybox
  command: ["sh", "-c", "mkdir -p /data/reports && chmod -R 777 /data/reports"]

containers:
- envFrom:
  - configMapRef:
      name: datananite-config
  - secretRef:
      name: datananite-secrets
  resources:
    requests: {memory: "512Mi", cpu: "250m"}
    limits:   {memory: "1Gi",  cpu: "500m"}
  livenessProbe:
    initialDelaySeconds: 90         # ← IMPORTANT: 90s needed (LangChain cold start)
  readinessProbe:
    initialDelaySeconds: 90
```

---

## PVC Design

Each service has its **own dedicated PVC** (ReadWriteOnce). This is critical — sharing PVCs between services caused `Multi-Attach` errors when pods landed on different nodes.

| PVC Name | Size | Used By |
|----------|------|---------|
| agent-api-pvc | 5Gi | agent-api |
| ontology-api-pvc | 5Gi | ontology-api |
| kg-api-pvc | 5Gi | kg-api |
| dialog-api-pvc | 5Gi | dialog-api |
| conformity-api-pvc | 5Gi | conformity-api |
| chat-ui-pvc | 5Gi | chat-ui (also uses metadata-catalog-pvc) |
| tech-ui-pvc | 5Gi | tech-ui |
| metadata-catalog-pvc | 5Gi | chat-ui (for /data — metadata.db, kg_store.db) |
| unstructured-pvc | 10Gi | unstructured-api |

---

## Environment Variables

### ConfigMap (datananite-config) — Non-sensitive
```
LOG_LEVEL=info
DIALOG_ENV=production
UNSTRUCTURED_WORKERS=4
DATA_DIR=/data/reports
GLOSSARY_DB=/data/reports/glossary.db
METADATA_DB=/data/metadata.db
KG_STORE_DB=/data/kg_store.db
UNSTRUCTURED_DB=/data/unstructured.db
METADATA_API_URL=http://agent-api-service:8000
ONTOLOGY_API_URL=http://ontology-api-service:8001
KG_API_URL=http://kg-api-service:8002
DIALOG_API_URL=http://dialog-api-service:8003
CONFORMITY_API_URL=http://conformity-api-service:8004
SHACL_API_URL=http://shacl-api-service:8007
UNSTRUCTURED_API_URL=http://unstructured-api-service:8008
UNSTRUCTURED_PUBLIC_URL=http://datananite.endpoints.cog01k24f1ea555zdv7ynzthxanz5.cloud.goog/api/unstructured
```

### Secret (datananite-secrets) — Sensitive
```
ANTHROPIC_API_KEY=<real key needed>
```

To update ANTHROPIC_API_KEY in cluster:
```bash
kubectl create secret generic datananite-secrets \
  --namespace=datananite \
  --from-literal=ANTHROPIC_API_KEY=<your-real-key> \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## CI/CD Pipeline Summary

**File:** `.github/workflows/deploy.yml`

**Trigger:** Push to `main` branch OR manual `workflow_dispatch`

**Steps:**
1. Checkout code
2. Auth to GCP (`GCP_SA_KEY` secret)
3. Install gke-gcloud-auth-plugin
4. Configure Docker for GCR
5. Build all 10 Docker images (tagged with `${{ github.sha }}` + `latest`)
6. Push all 10 images to `gcr.io/cog01k24f1ea555zdv7ynzthxanz5/datananite-*`
7. Get GKE credentials
8. Update `datananite-secrets` with `ANTHROPIC_API_KEY`
9. Apply ConfigMap and PVCs
10. `kubectl set image` for all 10 deployments (SHA-tagged)
11. `kubectl rollout restart` all 10 deployments
12. `kubectl rollout status` verify all 10 (5 min timeout each)
13. Print final pod/service/ingress/PVC status

---

## Useful kubectl Commands

```bash
# Connect to cluster
gcloud container clusters get-credentials lvp-fastapi-cluster \
  --zone asia-south1-a --project cog01k24f1ea555zdv7ynzthxanz5

# Check all pods
kubectl get pods -n datananite

# Check all resources
kubectl get all -n datananite

# View logs for a service
kubectl logs -n datananite -l app=agent-api --tail=50

# Check ingress
kubectl get ingress -n datananite

# Check PVCs
kubectl get pvc -n datananite

# Update ANTHROPIC_API_KEY
kubectl create secret generic datananite-secrets \
  --namespace=datananite \
  --from-literal=ANTHROPIC_API_KEY=<key> \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart a specific service
kubectl rollout restart deployment/agent-api -n datananite

# Restart all services
kubectl rollout restart deployment -n datananite

# Check events (useful for debugging)
kubectl get events -n datananite --sort-by=.lastTimestamp
```

---

## Issues Encountered & Solutions

| Issue | Cause | Solution Applied |
|-------|-------|-----------------|
| `PermissionError: /data/reports/uploads` | Container runs as non-root, can't write to mounted PVC | Added `initContainer` with busybox to `chmod -R 777 /data/reports` |
| `Multi-Attach error` on PVCs | Multiple pods on different nodes tried to mount same `ReadWriteOnce` PVC | Split shared `reports-pvc` into individual PVCs per service |
| Pods stuck in `ContainerCreating` | Old pods (from rolling update) held PVCs, new pods couldn't attach | Changed deployment strategy from `RollingUpdate` to `Recreate` |
| `CrashLoopBackOff` on agent-api | Pods were healthy but display showed old state | Was display lag — pods were actually running fine |

---

## What to Do Next (For Continuing Claude Code Session)

1. **Verify GitHub Secrets are added** — check `GCP_SA_KEY` and `ANTHROPIC_API_KEY` exist
2. **Merge PR** `feat/gke-deployment` → `main`
3. **Monitor CI/CD** at `https://github.com/krishnarayapudi2571/Datananite/actions`
4. **Verify all pods running** after CI/CD: `kubectl get pods -n datananite`
5. **Test the app** at `http://datananite.endpoints.cog01k24f1ea555zdv7ynzthxanz5.cloud.goog`
6. **Optional future work:**
   - Add TLS/HTTPS (Google-managed certificate)
   - Add HorizontalPodAutoscaler for high-traffic services
   - Add PodDisruptionBudgets (pdb.yaml not yet created for DataNanite)
   - Consider Filestore (NFS) if shared storage between services is needed
   - Add MongoDB authentication (F-004 equivalent)
