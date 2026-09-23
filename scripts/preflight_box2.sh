#!/usr/bin/env bash
# Preflight part 2: Qwen3-8B fit test (HF, the extraction path) and a vLLM venv with a
# throughput smoke test (the generation path). Log: ~/preflight2.log
set -uo pipefail
exec > >(tee -a ~/preflight2.log) 2>&1
export PATH="$HOME/.local/bin:$PATH"
REPO=~/agent-failure-probes

echo "===== 8B fit (HF bf16, sdpa)"
$REPO/.venv/bin/python - <<'EOF'
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
rev = "b968826d9c46dd6066d109eabc6255188de91218"
t = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=rev)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
text = tok.apply_chat_template([{"role": "user", "content": "Say hi."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
out = m.generate(ids, max_new_tokens=8, do_sample=False)
# replay-style forward at a realistic long context (8k) to size extraction memory
long = torch.randint(100, 20000, (1, 8192), device="cuda")
with torch.inference_mode():
    h = m.model(input_ids=long, use_cache=False, output_hidden_states=True).hidden_states
print("8B gen:", repr(tok.decode(out[0, ids.shape[1]:])), "layers", len(m.model.layers), "hidden_states", len(h),
      "peak GiB %.1f" % (torch.cuda.max_memory_allocated() / 2**30), "s %.0f" % (time.time() - t))
print("PREFLIGHT OK   qwen3-8b fits (gen + 8k-token all-layer forward)")
EOF

echo "===== vLLM venv"
E=~/envs/vllm; [ -d $E ] || uv venv -q $E --python 3.12
VIRTUAL_ENV=$E uv pip install -q vllm && $E/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" \
  && echo "PREFLIGHT OK   vllm install" || echo "PREFLIGHT FAIL vllm install"

echo "===== vLLM throughput (Qwen3-1.7B, seeded sampling, 512 prompts)"
$E/bin/python - <<'EOF'
import time
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", dtype="bfloat16",
          gpu_memory_utilization=0.85, max_model_len=8192, enable_prefix_caching=True, seed=0)
tok = llm.get_tokenizer()
p = tok.apply_chat_template([{"role": "user", "content": "Craft a wooden pickaxe in Minecraft. List the steps as actions."}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
params = [SamplingParams(temperature=0.8, top_p=0.95, top_k=-1, max_tokens=256, seed=1000 + i) for i in range(512)]
t = time.time(); outs = llm.generate([p] * 512, params); dt = time.time() - t
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
# determinism: same seed twice -> same tokens
a = llm.generate([p], [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64, seed=7)])[0].outputs[0].token_ids
b = llm.generate([p, p], [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64, seed=7)] * 2)
print(f"vLLM 1.7B: {ntok} tokens in {dt:.1f}s = {ntok/dt:.0f} tok/s; per-request seed reproducible:",
      list(a) == list(b[0].outputs[0].token_ids) == list(b[1].outputs[0].token_ids))
print("PREFLIGHT OK   vllm throughput")
EOF
df -h ~ | tail -1
echo "PREFLIGHT2 DONE $(date -u)"
