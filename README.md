# Getting Started with vLLM on CPU

**Estimated Time:** 20-25 minutes

---

## Owner

**Name:** Daina Bouquin
**GitHub:** [@dbouquin](https://github.com/dbouquin)

---

## What This Tutorial Teaches

This tutorial teaches you how to serve and query large language models locally using vLLM's CPU backend, without requiring GPU hardware.

---

## Who This Is For

**Target Audience:** Python developers and data scientists who want to experiment with LLM inference locally without GPU access.

**Prerequisites:**

**Knowledge prerequisites:**
- Familiar with Python basics and command-line operations
- Understanding of basic API concepts (making HTTP requests)

**Installation prerequisites:**
- conda or mamba installed (see [install guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))
- At least 8GB of RAM available

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Configure** a vLLM server to run on CPU for local LLM inference
2. **Deploy** a small language model using vLLM's OpenAI-compatible API
3. **Query** the model programmatically using Python requests

---

## Quick Start

### 1. Clone this repository

```bash
cd /Users/dbouquin/Documents/vllm-cpu-guide
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate vllm-cpu-demo
```

### 3. Run the tutorial script

```bash
python tutorial.py
```

### 4. Verify your setup

The script will start a vLLM server and query it. You should see:
```
✓ vLLM server starting on http://localhost:8000
✓ Model loaded: facebook/opt-125m
✓ Response received from model
```

---

## External Dependencies

This tutorial uses:

| Service | Type | Fallback | Notes |
|---------|------|----------|-------|
| Hugging Face Hub | External with fallback | Model cached locally after first download | Internet required for first run only |

---

## Tutorial Sections

### Section 1: Environment Setup

**Starting state:** Before starting this section, you should have:
- Conda environment activated (`vllm-cpu-demo`)
- Terminal open in the project directory

**What you'll do:**
- Verify vLLM installation
- Understand CPU backend configuration

**Steps:**

1. **Verify vLLM is installed**

   Check that vLLM is available in your environment:

   ```bash
   python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
   ```

2. **Understand the CPU backend**

   vLLM automatically detects when no GPU is available and falls back to CPU inference. For this tutorial, we'll use a small model (125M parameters) that runs efficiently on CPU.

**Checkpoint:** At the end of this section, you should see:
```
vLLM version: 0.4.2
```

If you see: `ModuleNotFoundError: No module named 'vllm'`
**Solution:** Ensure you activated the conda environment: `conda activate vllm-cpu-demo`

---

### Section 2: Starting the vLLM Server

**Starting state:** Before starting this section, you should have vLLM installed and verified.

**What you'll do:**
- Launch a vLLM server with a small language model
- Configure the server for CPU-only inference

**Steps:**

1. **Start the server in the background**

   Open a new terminal, activate the environment, and run:

   ```bash
   conda activate vllm-cpu-demo
   python start_server.py
   ```

   This starts a vLLM server using the `facebook/opt-125m` model.

2. **Wait for the model to load**

   You'll see output indicating the model is downloading (first run) and loading into memory. This takes 1-2 minutes on CPU.

**Checkpoint:** You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If you see: `RuntimeError: CUDA not available`
**Solution:** This is expected! vLLM will automatically fall back to CPU. Ignore this message.

---

### Section 3: Querying the Model

**Starting state:** Before starting this section, you should have the vLLM server running on port 8000.

**What you'll do:**
- Send API requests to the running model
- Parse and display model responses

**Steps:**

1. **Send a test query**

   In a new terminal window (keep the server running), run:

   ```bash
   conda activate vllm-cpu-demo
   python query_model.py
   ```

2. **Examine the response**

   The script sends a prompt to the model and prints the generated text.

**Checkpoint:** You should now see output like:
```
Prompt: "Once upon a time"
Response: "Once upon a time, there was a small village..."
Tokens generated: 50
Time taken: ~2.3 seconds
```

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

### Mistake 1: Using models that are too large for CPU

❌ **Don't do this:**
```python
# This will be extremely slow or crash on CPU
model = "meta-llama/Llama-2-70b"
```

✅ **Do this instead:**
```python
# Use small models optimized for CPU inference
model = "facebook/opt-125m"  # 125M parameters
# or
model = "gpt2"  # 124M parameters
```

**Why:** Large models (>1B parameters) are impractical on CPU due to memory and speed constraints.

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
time.sleep(30)  # Wait for model to load
query_model()
```

**Why:** Model loading takes time, especially on CPU. The server needs to fully initialize before accepting requests.

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

### Issue: Model generation is very slow

**Symptoms:** Each token takes >1 second to generate

**Solution:** This is expected on CPU! For faster inference:
- Use smaller models (opt-125m, gpt2)
- Reduce max_tokens in your queries
- Consider using a GPU-enabled machine for production workloads

**Why this happens:** CPU inference is significantly slower than GPU. This tutorial prioritizes accessibility over speed.

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
