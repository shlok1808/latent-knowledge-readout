"""Offline tiny model + tokenizer for the syco tests (no network, CPU, seconds to build).

Import from a test file as ``from syco_tiny import tiny_tokenizer, tiny_model`` (pytest puts
``tests/`` on ``sys.path``; ``tests.syco_tiny`` also works when running ``python -m pytest`` from
the repo root).

Tokenizer (``tiny_tokenizer()``):
- A BPE model over a GPT-2-style regex pre-tokenizer that keeps a leading space on words, so
  "A".."E" and " A".." E" are each ONE token (like Llama 3's tiktoken vocab), and so are a few
  words that appear in the prompts ("User", " answer", ...). Anything else falls back to single
  characters; characters outside the alphabet map to ``<unk>``.
- Special tokens: ``<pad>`` (pad), ``<s>`` (bos, prepended when ``add_special_tokens=True``),
  ``</s>`` (eos), ``<unk>``, and role markers ``<|system|>``, ``<|user|>``, ``<|assistant|>``.
- ``chat_template`` renders ``<s>`` itself (like Llama 3's template), then one
  ``<|role|>\\n{content}\\n`` block per message, and ``<|assistant|>\\n`` with
  ``add_generation_prompt=True``; the answer letter is the next token. So native prompts must be
  tokenized with ``add_special_tokens=False`` or they get a double BOS.
- ``tiny_tokenizer(chat_template=False)`` gives a tokenizer without a chat template (like the base
  model); ``tiny_tokenizer(pad=False)`` gives one without a pad token (to test the eos fallback).

Model (``tiny_model(tokenizer)``): ``LlamaForCausalLM`` with 2 layers, hidden 32, 2 heads,
vocab = len(tokenizer), float32, eval mode, weights seeded (``seed=0`` by default), so outputs are
deterministic across calls. Untrained: its letter probabilities are arbitrary but fixed.
"""
import string

import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]
CHARS = list(string.ascii_letters + string.digits + " \n.,:;?!'\"()-%")
# Words merged into single tokens (a leading space is part of the word, as in byte-level BPE).
WORDS = [
    " A", " B", " C", " D", " E",
    "User", "Assistant", "System", " answer", " the", " is", " I", " believe", " correct",
    " Give", " your", " final", " using", " exactly", " one", " option", " letter",
    " Please", " question", " again", " think", " don", " What", " Which",
]
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)
# GPT-2-style split: optional leading space + letters / digits / punctuation, or whitespace runs.
SPLIT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"


def _vocab_and_merges():
    vocab = {}
    for token in SPECIALS + CHARS:
        vocab.setdefault(token, len(vocab))
    merges = []
    for word in WORDS:
        for i in range(1, len(word)):
            left, right, merged = word[:i], word[i], word[: i + 1]
            if merged not in vocab:
                vocab[merged] = len(vocab)
                merges.append((left, right))
    return vocab, merges


def tiny_tokenizer(chat_template=True, pad=True):
    vocab, merges = _vocab_and_merges()
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(SPLIT), behavior="isolated")
    backend.decoder = decoders.Fuse()
    backend.add_special_tokens(SPECIALS)
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", vocab["<s>"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>" if pad else None,
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    if chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def tiny_model(tokenizer, seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    # Default init is tiny (std 0.02), which makes every next-token distribution near uniform.
    # Scale weights up so letter probabilities and position effects are clearly non-trivial.
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0.0, 0.3)
    model.eval()
    return model
