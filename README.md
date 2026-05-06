# Getting Started with vLLM on GPU

[![Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Anaconda-Labs/vllm-cpu-guide/main/.github/badges/status.json?1778105693&cacheSeconds=300)](https://github.com/Anaconda-Labs/vllm-cpu-guide)


**Estimated Time:** 15-20 minutes

---

## Owner

**Name:** Daina Bouquin
**GitHub:** [@dbouquin](https://github.com/dbouquin)

---

## Learning Objectives

This tutorial teaches you how to serve and query large language models using vLLM on GPU infrastructure, specifically designed for Outerbounds GPU workstations.

---

## Who This Is For

**Target Audience:** Python developers and data scientists who want to deploy high-performance LLM inference using vLLM on GPU infrastructure.

**Prerequisites:**

**Knowledge prerequisites:**
- Familiar with Python basics and command-line operations
- Understanding of basic API concepts (making HTTP requests)
- Basic familiarity with remote development environments

**Installation prerequisites:**
- Access to an Outerbounds GPU workstation or similar Linux GPU instance
- conda or mamba installed (see [install guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))
- At least 8GB GPU VRAM (for the example model)

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Deploy** vLLM server on GPU infrastructure for fast LLM inference
2. **Configure** a language model using vLLM's OpenAI-compatible API
3. **Query** the model programmatically and measure GPU performance benefits

---

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/Anaconda-Labs/vllm-cpu-guide.git
cd vllm-cpu-guide
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate vllm-gpu-demo
```

### 3. Verify GPU availability

```bash
nvidia-smi
```

You should see your GPU listed. If not, ensure you're on a GPU-enabled workstation.

### 4. Run the tutorial script

```bash
python tutorial.py
```

### 5. Verify your setup

The script will check your GPU and provide instructions. You should see:
```
✓ vLLM is installed
✓ GPU detected: NVIDIA A100 (or similar)
✓ CUDA available
```

---

## External Dependencies

This tutorial uses:

| Service | Type | Fallback | Notes |
|---------|------|----------|-------|
| Hugging Face Hub | External with fallback | Model cached locally after first download | Internet required for first run only |
| NVIDIA GPU | Required | None | Outerbounds workstations or similar GPU infrastructure |

---

## Tutorial Sections

### Section 1: Environment Setup

**Starting state:** Before starting this section, you should have:
- Conda environment activated (`vllm-gpu-demo`)
- Terminal open in the project directory
- GPU-enabled workstation (Outerbounds or similar)

**What you'll do:**
- Verify vLLM and GPU availability
- Check CUDA installation

**Steps:**

1. **Verify GPU is accessible**

   Check that your GPU is visible:

   ```bash
   nvidia-smi
   ```

   You should see output showing your GPU (e.g., A100, V100, T4).

2. **Verify vLLM is installed**

   Check that vLLM is available:

   ```bash
   python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
   ```

3. **Check CUDA availability**

   Verify PyTorch can access the GPU:

   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

**Checkpoint:** At the end of this section, you should see:
```
vLLM version: 0.6.0
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB (or similar)
```

If you see: `CUDA available: False`
**Solution:** Ensure you're on a GPU workstation and CUDA toolkit is properly installed

---

### Section 2: Starting the vLLM Server

**Starting state:** Before starting this section, you should have vLLM installed, GPU verified, and CUDA available.

**What you'll do:**
- Launch a vLLM server with GPU acceleration
- Configure the server to use your GPU

**Steps:**

1. **Start the server**

   Open a new terminal, activate the environment, and run:

   ```bash
   conda activate vllm-gpu-demo
   python start_server.py
   ```

   This starts a vLLM server using the `facebook/opt-125m` model with GPU acceleration.

2. **Monitor GPU usage (optional)**

   In another terminal, watch GPU utilization:

   ```bash
   watch -n 1 nvidia-smi
   ```

   You'll see GPU memory usage increase as the model loads.

3. **Wait for the model to load**

   You'll see output indicating the model is downloading (first run) and loading into GPU memory. This takes 30-60 seconds with GPU.

**Checkpoint:** You should see:
```
INFO:     Started server process
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Using GPU: NVIDIA A100-SXM4-40GB
```

If you see: `RuntimeError: CUDA not available`
**Solution:** Check that you're on a GPU workstation and run `nvidia-smi` to verify GPU access

---

### Section 3: Querying the Model

**Starting state:** Before starting this section, you should have the vLLM server running on port 8000 with GPU acceleration.

**What you'll do:**
- Send API requests to the GPU-accelerated model
- Measure GPU performance benefits

**Steps:**

1. **Send a test query**

   In a new terminal window (keep the server running), run:

   ```bash
   conda activate vllm-gpu-demo
   python query_model.py
   ```

2. **Observe GPU acceleration**

   The script sends prompts to the model and measures response time. With GPU acceleration, generation is significantly faster than CPU.

**Checkpoint:** You should now see output like:
```
Prompt: "Once upon a time"
Response: "Once upon a time, there was a small village..."
Tokens generated: 50
Time taken: ~0.3 seconds (GPU accelerated)
Tokens per second: ~166
```

Compare this to CPU inference which would take 2-5 seconds for the same task!

---

## Output Examples

Throughout this tutorial, you'll see outputs like these:

### Example 1: Server Startup

**Code:**
```bash
python start_server.py
```

**Expected Output:**
```
INFO: Loading model facebook/opt-125m
INFO: Model loaded successfully
INFO: Starting vLLM server on port 8000
```

### Example 2: API Query

**Code:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "facebook/opt-125m",
        "prompt": "The future of AI is",
        "max_tokens": 30
    }
)
print(response.json()["choices"][0]["text"])
```

**Expected Output:**
```
"The future of AI is bright, with applications in healthcare, education, and automation transforming how we work."
```

---

## Extension Challenges

Want to go further? Try these optional challenges:

- [ ] **Challenge 1:** Try a different small model like `gpt2` or `distilgpt2`
  - *Hint:* Change the model name in `start_server.py`
  - *Resource:* [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)

- [ ] **Challenge 2:** Add streaming responses to display tokens as they're generated
  - *Hint:* Use the `/v1/completions` endpoint with `stream=True`
  - *Resource:* [vLLM OpenAI API docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

- [ ] **Challenge 3:** Create a simple chatbot that maintains conversation context
  - *Why try this:* Learn about prompt engineering and context management

---

## Common Mistakes to Avoid

### Mistake 1: Using models larger than GPU memory

❌ **Don't do this:**
```python
# This will crash if your GPU has less than 80GB VRAM
model = "meta-llama/Llama-2-70b"
```

✅ **Do this instead:**
```python
# Match model size to your GPU memory
# For 40GB GPU (A100):
model = "meta-llama/Llama-2-13b"  # ~26GB VRAM
# For 16GB GPU (T4):
model = "facebook/opt-6.7b"  # ~13GB VRAM
# For this tutorial:
model = "facebook/opt-125m"  # ~500MB VRAM
```

**Why:** Models must fit in GPU VRAM. Check your GPU memory with `nvidia-smi`.

### Mistake 2: Forgetting to wait for server startup

❌ **Don't do this:**
```python
# Querying immediately after starting server
start_server()
query_model()  # This will fail!
```

✅ **Do this instead:**
```python
start_server()
time.sleep(10)  # Wait for model to load on GPU
query_model()
```

**Why:** Model loading takes time (even on GPU). The server needs to fully initialize before accepting requests.

---

## Troubleshooting

### Issue: Server won't start - "Address already in use"

**Symptoms:** Error message when starting server: `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9
# Then restart the server
python start_server.py
```

**Why this happens:** A previous server instance is still running on port 8000.

---

### Issue: Environment activation not working

**Solution:**
```bash
conda init bash  # or zsh, fish, etc.
# Restart your terminal
```

---

### Issue: Model generation is slower than expected

**Symptoms:** Each token takes >0.1 seconds to generate on GPU

**Solution:** Check these common causes:
- GPU memory is full (check `nvidia-smi`)
- Other processes are using the GPU
- Model is larger than optimal for your GPU
- Not enough GPU warming up time

**Why this happens:** vLLM needs exclusive GPU access for best performance. Close other GPU applications.

---

### Issue: CUDA out of memory error

**Symptoms:** Error message: `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Use a smaller model
# Or reduce max_model_len parameter in start_server.py
--max-model-len 2048  # Reduces memory usage
```

**Why this happens:** The model + context window exceeds available GPU memory.

---

## What's Next?

After completing this tutorial, you can:

- **Scale to GPU** - [vLLM GPU Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- **Try quantized models** - [Optimizing Model Size for CPU](https://huggingface.co/docs/transformers/main_classes/quantization)
- **Build a chatbot UI** - Combine vLLM with Gradio or Streamlit

### Related Tutorials
- [Building production LLM applications with vLLM](https://docs.vllm.ai/en/latest/)
- [Prompt engineering best practices](https://platform.openai.com/docs/guides/prompt-engineering)

---

## Glossary

**vLLM:** A high-throughput and memory-efficient inference engine for large language models, supporting both GPU and CPU backends.

**Token:** The basic unit of text processing in LLMs. A token is roughly equivalent to a word or word fragment.

**Inference:** The process of using a trained model to generate predictions or outputs (as opposed to training the model).

**OpenAI-compatible API:** An API specification that matches OpenAI's API format, allowing tools built for OpenAI to work with vLLM.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

Found an issue or want to improve this tutorial?
1. Open an issue describing the problem or improvement
2. Submit a pull request with your changes
3. Ensure any code changes are tested
