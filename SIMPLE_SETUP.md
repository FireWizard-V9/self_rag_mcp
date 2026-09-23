# Simple vLLM Disaggregated Setup (Official Docs)

**3 Instances. 3 Commands. ONE Dockerfile + ONE entrypoint.sh**

Uses environment variables to differentiate prefill (kv_producer) vs decode (kv_consumer).

---

## Hardware

| Instance | Type | GPU | Cost/month |
|----------|------|-----|-----------|
| Prefill | g4dn.xlarge | T4 16GB | $494 |
| Decode | g4dn.xlarge | T4 16GB | $494 |
| Router/LMCache | t3.medium | None | $36 |
| **Total** | | | **$1,024** |

---

## Instance Setup

### Instance 1: Prefill (g4dn.xlarge)

```bash
# Prerequisites
sudo apt-get install -y nvidia-driver-550 nvidia-container-toolkit
sudo systemctl restart docker

# Pre-cache model
mkdir -p ~/.cache/huggingface

# Build the image (one-time)
cd self_rag_retrieval
docker build -t vllm-server:latest ./vllm

# Run Prefill with KV Producer role
docker run -d \
  --name vllm-prefill \
  --gpus all \
  -e VLLM_KV_ROLE=kv_producer \
  -e GPU_MEMORY_UTILIZATION=0.80 \
  -e MAX_NUM_SEQS=32 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-server:latest
```

**Verify:**
```bash
docker logs -f vllm-prefill
curl http://localhost:8000/health
```

---

### Instance 2: Decode (g4dn.xlarge)

```bash
# Prerequisites (same as Instance 1)
sudo apt-get install -y nvidia-driver-550 nvidia-container-toolkit
sudo systemctl restart docker

mkdir -p ~/.cache/huggingface

# Build image (if not done on Instance 1)
cd self_rag_retrieval
docker build -t vllm-server:latest ./vllm

# Run Decode with KV Consumer role
docker run -d \
  --name vllm-decode \
  --gpus all \
  -e VLLM_KV_ROLE=kv_consumer \
  -e GPU_MEMORY_UTILIZATION=0.80 \
  -e MAX_NUM_SEQS=128 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-server:latest
```

**Verify:**
```bash
docker logs -f vllm-decode
curl http://localhost:8000/health
```

---

### Instance 3: Router + LMCache (t3.medium)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start Qdrant (for Self-RAG retrieval)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest

# Start vLLM Disaggregated Serving Proxy
# IMPORTANT: Replace IPs with Instance 1 and Instance 2 private IPs
PREFILL_IP="10.0.1.10"    # Instance 1
DECODE_IP="10.0.2.10"     # Instance 2

docker run -d \
  --name vllm-router \
  -e PYTHONUNBUFFERED=1 \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  python -m vllm.entrypoints.disaggregated_serving \
    --prefill-host ${PREFILL_IP} \
    --prefill-port 8000 \
    --decode-host ${DECODE_IP} \
    --decode-port 8000 \
    --host 0.0.0.0 \
    --port 8000 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ

# Verify
docker logs -f vllm-router
curl http://localhost:8000/health
curl http://localhost:6333/health
```

---

## Test End-to-End

From Instance 3 (Router):

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "prompt": "What is machine learning?",
    "max_tokens": 100
  }'
```

Should work: Prefill → NIXL transfer → Decode → Response

---

## Update Self-RAG MCP

Point to the router (Instance 3):

```bash
# .env
VLLM_BASE_URL=http://<INSTANCE_3_IP>:8000/v1
QDRANT_URL=http://<INSTANCE_3_IP>:6333
```

Restart your MCP server.

---

## How It Works

**Same Dockerfile. Same entrypoint.sh. Different environment variables.**

1. Build image once: `docker build -t vllm-server:latest ./vllm`
2. Run on Instance 1 with: `-e VLLM_KV_ROLE=kv_producer` (Prefill)
3. Run on Instance 2 with: `-e VLLM_KV_ROLE=kv_consumer` (Decode)
4. The entrypoint.sh script reads `VLLM_KV_ROLE` and adds appropriate NIXL flags

The entrypoint automatically configures:
- `kv_producer` → adds `--kv-connector NixlConnector --kv-role kv_producer`
- `kv_consumer` → adds `--kv-connector NixlConnector --kv-role kv_consumer`
- (default) → runs as standalone/monolithic

---

## Troubleshooting

### NIXL transfer not happening

```bash
# Check env vars on instances
docker exec vllm-prefill env | grep VLLM_KV
docker exec vllm-decode env | grep VLLM_KV

# Should show:
# VLLM_KV_CONNECTOR=NixlConnector
# VLLM_KV_ROLE=kv_producer (or kv_consumer)
```

### Decode can't reach Prefill

```bash
# From Instance 2, test Instance 1
docker run --rm curlimages/curl curl http://10.0.1.10:8000/health

# Should return JSON (not timeout)
```

### Out of Memory on T4

```bash
# Reduce sequences
--max-num-seqs 16  # Was 32/128

# Or reduce context
--max-model-len 4096  # Was 8192

# Or reduce GPU util
--gpu-memory-utilization 0.70  # Was 0.80
```

---

## Expected Performance

| Metric | Value |
|--------|-------|
| TTFT | 300-500ms |
| ITL | 50-80ms |
| Total (200 tokens) | 10-16s |
| Throughput | 0.06-0.08 req/sec |

**Note:** No throughput improvement Phase 1 (validation only). Improvement comes Phase 2 (multiple decoders).

---

## Reference

- **Official vLLM Disaggregated Serving:** https://docs.vllm.ai/en/latest/examples/disaggregated/disaggregated_serving/
- **NIXL Connector:** https://docs.vllm.ai/en/latest/features/nixl_connector_usage/

---

**Status:** Simple, Official, Ready ✓
