# One-time: seed the embedding model onto the shared PVC

`dialog-api`, `orchestrator-api`, and `chat-ui` all mount `datananite2-shared-pvc`
at `/data` (see `deployment/k8s/*.yaml`). `SENTENCE_TRANSFORMER_MODEL_PATH` in
`deployment/k8s/configmap.yaml` points at `/data/models/all-MiniLM-L6-v2` on
that volume. `dialog_agent/embedding_cache.py` and `orchestrator_api.py`'s
`_gr_sentence_transformer()` load the model from that local path instead of
resolving it through the HF hub cache — so once the file is on the PVC, no
pod or CI build ever needs to reach huggingface.co again, and the model
survives every future redeploy without a rebuild.

`kubectl cp` streams through the Kubernetes API server, not through the
target pod's own network — so this works even though the pods themselves
have no egress to huggingface.co. You only need `kubectl` access to the
cluster and, on your OWN machine, internet access to huggingface.co (the
cluster's network restrictions don't apply to your machine).

## Steps (run once)

1. On a machine with internet access (not the CI runner, not a pod):
   ```
   pip install sentence-transformers
   python scripts/download_embedding_model.py
   ```
   This writes `models/all-MiniLM-L6-v2/` locally (~90MB). This folder is
   NOT committed to the repo — it's a one-time transfer artifact.

2. Copy it onto the PVC via any currently-running pod that mounts `/data`
   (orchestrator-api, dialog-api, or chat-ui — pick whichever is up):
   ```
   kubectl get pods -n datananite2 -l app=orchestrator-api
   kubectl cp models/all-MiniLM-L6-v2 \
     datananite2/<pod-name>:/data/models/all-MiniLM-L6-v2
   ```

3. Verify it landed:
   ```
   kubectl exec -n datananite2 <pod-name> -- ls -la /data/models/all-MiniLM-L6-v2
   ```

4. Restart the three deployments so they pick up the ConfigMap path (if this
   is the first time `SENTENCE_TRANSFORMER_MODEL_PATH` is being added, a
   plain rollout restart is enough — no rebuild needed):
   ```
   kubectl rollout restart deployment/dialog-api -n datananite2
   kubectl rollout restart deployment/orchestrator-api -n datananite2
   kubectl rollout restart deployment/chat-ui -n datananite2
   ```

5. Confirm the fix took: check logs for
   `retrieve_node: cache built (sentence-transformers)` instead of
   `retrieve_node: embedding build failed (...) — falling back to keyword`.

Re-run only if the PVC is ever recreated (e.g. cluster migration) — normal
redeploys, pod restarts, and image rebuilds all keep using the existing file
on the volume.
