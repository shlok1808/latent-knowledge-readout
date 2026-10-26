# Neuron-level interpretability (Cammarata, "Neural Networks Are Already Clean") and what it enables for knowledge-vs-action divergence

Research date: 2026-10-16, one day after the essay was published.

**How the sources were read.** The primary essay (https://already-clean.goodfire.com, about 171k characters) was read through WebFetch, which summarizes the page with a small model. I covered characters 0–100k and 130k–end. I did not read the 100k–130k section, which is mostly the SAE/vision appendix. Short phrases quoted from the essay came through that extraction step, so check the wording against the page before quoting it in print. I could not fetch the X posts directly (HTTP 402), so the reactions come from news articles that quote them. The arXiv API returned HTTP 429, so I read paper abstracts from arxiv.org/abs pages instead. None of the related papers in the section on related work (2604.19117, 2507.11878, 2512.01797, 2604.19765, 2601.18939, 2406.14144, EACL-2026 safety knowledge neurons) appear in `papers_master.csv`. The CSV already lists 2608.00218, 2607.00158 and NEAT.

---

## Q1. What exactly does the essay claim, how was the audit done, and is a technical paper out?

### Takeaway
The essay is an explicitly preliminary "working hypothesis". The Qwen3-32B figures are about 90% of neuron *poles* with one main job, about 7% with two related jobs, and about 2–4% not yet understood. They come from "preliminary audits", and the essay does not describe the protocol: no sample size, layers, thresholds or reviewer process are given. Most of the rigorous evidence is in vision models (InceptionV1 and ConvNeXt). The language-model experiments are deferred to a forthcoming technical companion paper, which is not yet out, plus a separate Kimi K3 paper.

### Cited Findings
**Headline numbers and how they are hedged**
- "In preliminary audits of Qwen3-32B, we estimate that roughly 90% of neuron poles have one main job" (often with "small related offshoots"). The essay adds "~7% have two related jobs" and "only ~2-4% are not yet understood". The unit is the neuron *pole*: the positive and negative sides of each neuron are scored separately. — [Essay](https://already-clean.goodfire.com)
- Looking only at strong activations gives "closer to 97% non-arbitrary mixtures, but these are relatively small samples." — [Essay](https://already-clean.goodfire.com)
- The author says the share depends on four choices: the weak-activation threshold, how deep into sub-circuits you go, how dense those sub-circuits are, and what counts as "one thing". He calls the number "a broad pointer to the regime", not a precise metric. He also says he is "somewhat against trying to standardize things across all neurons" and prefers each neuron's natural sparsity and raw activation shape to percentiles. — [Essay](https://already-clean.goodfire.com)
- The baselines are Towards Monosemanticity, where the median neuron scored 0 of 14 on its rubric, and OpenAI's 2023 automated interpretability work, which found about 1.7% of neurons interpretable "by our most generous measure". Cammarata co-authored the OpenAI work. — [Essay](https://already-clean.goodfire.com)
- The 1.7% figure comes from different models (GPT-2-era) and different tools, so the comparison with 90% is not like-for-like. — [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/)

**Argument structure**
- The essay makes four numbered claims:
  1. Networks use one learning principle, not two ("or nearly enough that it doesn't matter").
  2. With modern models and tools, the resulting science goes better than expected.
  3. Understanding compounds, so a network's own mechanisms should become part of the toolkit.
  4. Interpretability "might be far more tractable, possibly in the near term". The essay marks this step "Highly speculative".

  It frames the whole argument as a "bitter lesson" for interpretability: scaling ordinary investigation beats engineering a new decomposition. — [Essay](https://already-clean.goodfire.com)

**Recursive generalization**
- As a neuron learns, it "sprouts offshoots the way a tree branch sprouts sub-branches". Examples:
  - "iy" → "diy" → "do it yourself", which contains no "iy".
  - Geology → Cesium, the JavaScript library.
  - InceptionV1 rapeseed fields → tennis balls → dogs → broccoli.

  The essay says this was first hinted at as "conjoined features" in the Multimodal Neurons work. It claims recursive generalization predicts nearly the opposite of superposition, and that this can be tested by watching neurons grow over training. — [Essay](https://already-clean.goodfire.com)
- The essay defines superposition narrowly: "unrelated things are placed together because they don't interfere". It separates this from the looser sense of "more features than dimensions" and calls superposition "the last hypothesis" to reach for. — [Essay](https://already-clean.goodfire.com)

**Earlier polysemanticity explained as tooling and architecture**
- *Cat/car neuron 4e:55.* Reread as a V-shape detector: pointed shapes (flames, fox faces, car fronts) push it positive and round shapes push it negative. — [Essay](https://already-clean.goodfire.com)
- *Car→dog circuit in "Zoom In".* Reread as shared pose. The essay says feature visualization was reward-hacked toward dogs. — [Essay](https://already-clean.goodfire.com)
- *Architecture.* InceptionV1's late layers "overflowed". Adding residual connections and LayerNorm gave clean layers. The last-layer interpretable share rose from about 5% with old tools, to about 50% with better tools, to about 70% after retraining with modern regularization, to about 99% in a same-size ConvNeXt. — [Essay](https://already-clean.goodfire.com)
- *Compressed layer.* InceptionV1 layer 3b was cut from 480 to 10 neurons, reaching 58.35% top-1 accuracy. The neurons became *coarser* (broad curve neurons), not superposed. A TopK SAE with 640 features explained 82% of variance, but splicing it in dropped top-1 accuracy from 58.8% to 43.0%. — [Essay](https://already-clean.goodfire.com)
- *Language examples.* A "Base64 + Bengali" neuron turned out to track "pr". A "comma neuron" tracked Colorado, with commas acting as a "summary motif" where attention collects. — [Essay](https://already-clean.goodfire.com)
- The essay states: "To my knowledge, no one has found a superposition circuit in a real network since my original one, which was an error." — [Essay](https://already-clean.goodfire.com), quoted in [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/)
- *Neurons vs SAEs.* The author trained SAEs on these layers and found no building blocks that the neurons lacked. About 88% of last-layer SAE activity is near-copies of single neurons. — [Essay](https://already-clean.goodfire.com)
- *Weak activations, vision only.* Ablating the weakest 75% of activations changed accuracy from 72% to 71%, while removing the strongest 7% dropped it to 4%. — [Essay](https://already-clean.goodfire.com)
- *Rare concepts.* Rare concepts are consolidated onto shared circuits rather than hidden features. In one example, teaching a model that an umbrella photo is a bee changed mostly the last three blocks within nine steps. — [Essay](https://already-clean.goodfire.com)

**What the language-model method actually is**
- The essay describes these tools:
  - Agents that "mapped neurons and circuits", built "explorable microscopes", tested explanations, and tracked "hundreds of thousands of attribution graphs".
  - Activations measured over a saved 100M-token corpus.
  - Steering tests, e.g. turning up the Colorado neuron makes the model want to go hiking.
  - Checks of upstream and downstream connections.
  - Ablations of neuron groups compared against random removal.
  - In one case, grouping neurons by an agent's descriptions.

  It criticizes rewarding explanations that "simulate activations" over explanations that are correct. That is a dig at the OpenAI/Transluce simulator-scoring paradigm. — [Essay](https://already-clean.goodfire.com)
- The language-model section is written with "Curt", and the essay says it draws on a "forthcoming technical paper" on language neurons. A separate companion paper will cover Kimi K3. — [Essay](https://already-clean.goodfire.com)
- Goodfire's research index lists only the essay (Oct 15, 2026) and no companion paper. — [Goodfire research](https://www.goodfire.com/research)

**Mentions of alignment**
- The only explicit alignment remark is that "'deception' is a useful initial frame, but the building blocks that contribute to deceptive behavior by models probably have alignment-relevant microstructure that will be hard to know without going in and looking." — [Essay](https://already-clean.goodfire.com)
- The essay lists known tool problems: a "double-negative regime in SwiGLU neurons" that a decomposition mixing many neurons together "would likely have hidden", and "an unread saliency knob that makes the residual stream hard to visualize". It says large models "aren't close to solved due to sheer size". — [Essay](https://already-clean.goodfire.com)

### Inferences
- The robust, quantified evidence is about vision. For LLMs, the 90/7/2–4 split is an unreproducible point estimate until the companion paper appears. Report it as "Goodfire's preliminary estimate", not as a finding.
- "Pole" as the unit roughly doubles the denominator and lets a neuron with unrelated positive and negative sides count as two clean poles. This is reasonable for SwiGLU neurons, which are signed, but it is not comparable to per-neuron metrics in OpenAI 2023 or Transluce 2024.
- The audit seems to focus on strong activations, since the essay says weak activations "add related things". Nanda's critique in Q2 targets exactly that region.

### Gaps
- Sample size, layers covered, threshold values, which agent or model wrote the explanations, and whether humans adjudicated "one main job": none of this is in the readable text. I may have missed it in the unread 100k–130k section, but the WebFetch passes over 0–100k found no protocol.
- I found no companion-paper preprint (arXiv or Goodfire) as of 2026-10-16.

---

## Q2. Reception: who agrees, who disputes, on what grounds; any replications?

### Takeaway
After one day the reception is mostly social media and press. The most substantive critique is from Neel Nanda (Google DeepMind), who says the essay is not "a significant update" and still expects superposition in the residual stream. Liv Gorton argues that branching neurons are compatible with superposition. I found no replications, counter-analyses, LessWrong or Alignment Forum thread, Hacker News thread, or response from Anthropic or Chris Olah.

### Cited Findings
- **Neel Nanda (GDM):** "This is interesting work, but I don't think it's a significant update to the big picture of interpretability. I'm on board with the broad idea of doing exploratory interpretability with neurons, e.g. neuron-based attribution graphs, for pragmatic problems…" — [Nanda on X](https://x.com/NeelNanda5/status/2110875974296654062) (the post text is visible in the search-result title; I could not fetch the full thread)
- Nanda's further points, as reported by the press:
  - He still expects models to represent more concepts than dimensions in the residual stream, and thinks the authors agree.
  - He does not see how *factual/memorized* content fits a monosemantic-neuron view. LLMs hold far more facts than neurons, probably partly in weak activations, which an audit of strong activations would miss.
  - He doubts that results from small vision models transfer to LLMs, because vision models memorize much less.
  - Explaining prominent activity would not give him confidence that nothing important was missed.
  - On the positive side, neurons are "just vastly easier to… do on a frontier model, which is great."

  — [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/); [officechai](https://officechai.com/ai/ai-researcher-nick-cammarata-who-helped-discover-superposition-now-says-the-theory-was-wrong-and-thats-great-news-for-understanding-ai/)
- **Liv Gorton** welcomed more legible networks. She argues that superposition just means holding more ideas than neurons, so recursive branching does not contradict it. — [implicator.ai, citing her X post](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/)
- **Roon (OpenAI, pseudonymous)** wrote that the work "may be remembered as a major moment in scientific history." The article notes he is a close friend of the author. — [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/)
- I found no reaction from Chris Olah or Anthropic. The only Olah statement that turned up is from 2023, when he called superposition "primarily an engineering problem". — [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/); [Olah on X, 2023](https://x.com/ch402/status/1709998674087227859)
- **No replications.** Outside researchers have not confirmed the results and the companion paper is unreleased, though Qwen3-32B is open, so anyone can check. — [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/)
- A press "claim check" also found "no independent reaction yet". It flags that Goodfire employs the author and sells interpretability tooling, so it recommends waiting for replication by groups without a stake in the outcome. — [explainx.ai](https://explainx.ai/blog/goodfire-neural-networks-are-already-clean-cammarata-neurons-interpretable-90-percent-qwen3-32b-2026)
- A conceptual counterpoint that predates the essay argues that superposition ≠ neuron polysemanticity: neurons can be polysemantic without superposition, and the reverse can also hold. This exact distinction is now in dispute. — [LessWrong: "Superposition is not just neuron polysemanticity"](https://www.lesswrong.com/posts/8EyCQKuWo6swZpagS/superposition-is-not-just-neuron-polysemanticity)
- **Adjacent evidence that partly supports Cammarata.** A June 2026 paper finds that the canonical toy model of "computation in superposition" (Compressed Computation, Braun et al. 2025) probably is not one: its gains came from an unintended mixing matrix in the labels. — [arXiv 2606.14673](https://arxiv.org/abs/2606.14673)

### Inferences
- The disagreement is narrower than the headlines suggest. Nanda and Gorton both locate superposition in the *residual stream / representational* sense, while Cammarata's claim is about *MLP-neuron polysemanticity* and superposition as a *learning principle*. Both sides could be partly right: MLP neurons can be mostly legible while residual-stream directions still exceed the dimension count.
- Critics have raised threshold dependence and explanation quality. Nobody has yet clearly raised the MLP-neuron vs residual-stream basis point in public. For the project this point matters most, because attention-mediated effects never pass through a neuron basis.

### Gaps
- No LessWrong, Alignment Forum or Hacker News thread was found by search. Threads may appear within days.
- Responses from Anthropic, Transluce, OpenAI interpretability and Apollo were not found.

---

## Q3. Related neuron-level work: which papers show knowledge neurons separate from action neurons?

### Takeaway
The neuron-level literature splits in two. Behavior-control neurons tend to be *sparse and steerable*: safety, sycophancy, tool misuse, and the essay's own "say not", hedge and "holds back" neurons. *Knowledge or truth* signals tend to be *distributed and readable but not controllable*: medical hallucination, and hallucination neurons that fail to transfer across domains. The cleanest knowledge-vs-action dissociations so far are at the level of attention heads and directions, not neurons:
- 2604.19117: heads carry "statement is wrong"; ablating them changes deference but not accuracy.
- 2507.11878: the harmfulness and refusal directions are separate and sit at different token positions.

The essay's own Qwen examples are the closest thing to *neuron-level* evidence of a separate "action/override" layer.

### Cited Findings
**Cammarata's own LLM examples (Qwen3-32B, neuron attribution graphs) that look like knowledge vs action**
- *Circuit Biology medical graph.* Anthropic's diagnostic graph, originally built for Claude 3.5 Haiku with transcoder features, was reproduced with Qwen3-32B MLP neurons, which need no reconstruction loss. "The diagnosis hub is just four neurons." Removing them makes the model ask about itching instead, as if moving to its next guess. — [Essay](https://already-clean.goodfire.com)
- *"Confident user".* About 25 "say not" neurons push back against the user's claim. Removing them lets Qwen go along and name Anna (P = 0.92). — [Essay](https://already-clean.goodfire.com)
- *"Budget leak".* About 19 hedge neurons push "most" and "likely". Removing them lets Qwen commit to Marcus. — [Essay](https://already-clean.goodfire.com)
- *Santa cookies.* Eight neurons whose *negative poles* relate to Santa carry the answer: removing them drops P(Santa) to 0.059 and Qwen blames the pet. Removing the twenty neurons *labeled* Christmas or Santa only lowers it to 76%. — [Essay](https://already-clean.goodfire.com)
- *Kimi K3 found-wallet case.* The question's facts feed lost-property and honesty neurons, which push toward "return". The biggest single push comes from neurons that "hold it back": removing one raises P(return) from 0.83 to 0.98. The graph is a hand-picked slice, and full audits at this scale "have not begun". — [Essay](https://already-clean.goodfire.com)

**Automated neuron description**
- *OpenAI 2023.* About 1.7% of neurons were interpretable by the most generous measure, as recalled in the essay. — [Essay](https://already-clean.goodfire.com)
- *Transluce, "Scaling Automatic Neuron Description" (Choi, Huang, Meng, Johnson, Steinhardt, Schwettmann; Oct 23, 2024).* Covered all 458,752 MLP neurons of Llama-3.1-8B-Instruct, both polarities. Descriptions were scored by simulator correlation. Automated descriptions averaged 0.61 vs 0.55 for human experts and beat humans on 75 of 100 neurons, at about $0.046 per neuron. The authors note that some neurons show "interesting relationships" between their two polarities but do not analyze this, and they mention "concept superposition" without testing it. — [Transluce](https://transluce.org/neuron-descriptions)

**Universal and output-regulation neurons**
- *"Universal Neurons in GPT2" (Gurnee et al., 2024).* 1–5% of neurons are universal across seeds. Universal functional roles include deactivating attention heads, changing next-token entropy, and predicting whether the next token is in a particular set (prediction and suppression roles). — [arXiv 2401.12181](https://arxiv.org/abs/2401.12181)
- *"Confidence Regulation Neurons" (Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda; NeurIPS 2024).* Entropy neurons write into the unembedding null space to rescale confidence through LayerNorm. Token-frequency neurons shift output toward or away from the unigram distribution. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e21955c93dede886af1d0d362c756757-Abstract.html)

**Truthfulness, hallucination and knowledge neurons**
- *"Truth Neurons" (Li, Cao, Yu, Suchow, Zhu; 2025).* Some neurons encode truthfulness regardless of subject matter. Suppressing those found with TruthfulQA hurts TruthfulQA and other benchmarks. — [arXiv 2505.12182](https://www.arxiv.org/abs/2505.12182v1)
- *H-Neurons (Gao et al., Tsinghua; Dec 2025).* Fewer than 0.1% of neurons predict hallucination. Intervening on them causally drives "over-compliance" behaviors, and they trace back to the base pre-trained model. — [arXiv 2512.01797](https://arxiv.org/abs/2512.01797). Secondary summaries say the compliance benchmarks were FalseQA, FaithEval, Sycophancy and Jailbreak; I have not verified this from the paper.
- *"Do Hallucination Neurons Generalize?" (Mar 2026).* H-neuron classifiers reach AUROC 0.783 within a domain but only 0.563 across domains, over 6 domains and 5 models of 3–8B. The authors conclude there is "no single universal neural signature". — [arXiv 2604.19765](https://arxiv.org/abs/2604.19765)
- *"Readable but Not Controllable" (Jun 2026).* Medical hallucination is detectable at AUROC 0.77–0.86, but the signal is distributed: a few hundred *random* neurons recover nearly all of it. Steering the top neurons does not reliably fix hallucination across 16 model-dataset combinations. — [arXiv 2607.00158](https://arxiv.org/abs/2607.00158)

**Sycophancy and deference**
- *"A Few Bad Neurons" (Jan 2026; also listed at NeurIPS 2025).* SAEs plus probes locate about 3% of MLP neurons most predictive of sycophancy. Fine-tuning only those neurons with gradient masking matches or beats SOTA on four sycophancy benchmarks (Gemma-2 2B/9B). — [arXiv 2601.18939](https://arxiv.org/abs/2601.18939)
- *"LLMs Know They're Wrong and Agree Anyway" (Pandey, Apr–May 2026).*
  - Across 12 open models, a few **attention heads** carry a "this statement is wrong" signal in both neutral and pressured contexts.
  - Silencing them sharply reduces sycophancy while factual accuracy is unchanged.
  - The same head-to-head paths drive sycophancy, factual lying and instructed lying.
  - For opinions, the same heads write to an orthogonal direction.
  - RLHF cut sycophantic behavior about 10× while the heads persisted or grew.

  — [arXiv 2604.19117](https://arxiv.org/abs/2604.19117)

**Refusal and safety**
- *"LLMs Encode Harmfulness and Refusal Separately" (Zhao, Huang, Wu, Bau, Shi; 2025).*
  - Harmfulness is a separate *direction* from refusal.
  - Steering refusal elicits refusals without changing the internal harmfulness judgment.
  - Harmfulness is encoded at the last instruction token and refusal at the last sequence token, so a model can judge an input harmless and still refuse it.

  — [arXiv 2507.11878](https://arxiv.org/abs/2507.11878)
- *"Finding Safety Neurons" (Chen et al., Tsinghua; 2024, v2 2025, NeurIPS 2025).* Patching about 5% of neurons restores about 90% of safety performance. Safety neurons and helpfulness neurons overlap heavily but use different activation patterns. — [arXiv 2406.14144](https://arxiv.org/abs/2406.14144)
- *"Unraveling LLM Jailbreaks Through Safety Knowledge Neurons" (Zhao, Ke, Huang; EACL 2026).* Projecting MLP neuron activations into vocabulary space shows a "Rejection" vs "Conformity" duality that emerges from about layer 10. Calibrating safety-knowledge-neuron activations (ActCali) gives a mean attack success rate above 97%, changing about 0.3% of parameters. SafeTuning, which fine-tunes only those neurons, reduces ASR. — [ACL Anthology](https://aclanthology.org/2026.eacl-long.83.pdf)

**Tool use**
- *PRISMS (Ke, Yin, Zhao, Huang; Jul 2026).* Failure-specific MLP neuron sets detect three failures:
  - Missing calls: 1–2 neurons, before generation, ROC-AUC 0.90–1.00.
  - Over-calling: 2–16 neurons, before generation.
  - Invalid arguments: about 128 neurons, from the generated call span, AUROC 0.86–0.90.

  These match dense residual baselines with 23–627× fewer features. The *same* neuron basis is used for steering: over-calling fell from 0.131 to 0.026 and tool-required accuracy rose by 14.2 points across 6 Qwen3/Llama/Gemma models. The abstract does not test whether detection neurons and causal neurons differ. — [arXiv 2608.00218](https://arxiv.org/abs/2608.00218)
- *Relation-specific neurons (EMNLP 2025).* Ablations find neurons tied to a relation rather than an entity. No single neuron fully encodes a fact, and deactivating one relation's neurons can sometimes *improve* recall of other relations. — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.52.pdf)

### Inferences
- **Knowledge and action are separable at the neuron level, but asymmetrically.**
  - "Action/override" components look sparse and steerable: say-not, hedge, holds-back, refusal/conformity, PRISMS over-calling and missing-call neurons.
  - "Knowledge" components either sit upstream as concept neurons (Lion King, Santa, diagnosis hub) or are distributed and redundant (truth and hallucination signals in 2607.00158 and 2604.19765).

  This asymmetry fits Nanda's critique that memorized knowledge is where neuron cleanliness is weakest.
- **Two strong dissociation results live outside the MLP neuron basis.** 2604.19117 uses attention heads and 2507.11878 uses directions at different token positions. A neuron-only analysis would see their downstream effects but not the routing that causes the divergence.
- **The Santa result is direct evidence for using negative poles.** Neurons *labeled* Santa mattered far less than neurons whose *negative pole* relates to Santa. Suppressed knowledge might likewise show up as negative-pole activity that top-activation descriptions (OpenAI, Transluce) miss.
- **"Detected, but steering fails" is itself a data point.** In H-neurons and medical hallucination, the signal is detectable but steering is unreliable. That is a "readable but not causal" signature, which separates knowledge that is *present but bypassed* from knowledge that is *present and suppressed*.

### Gaps
- I found no paper that explicitly separates, *within one model and one prompt*, a set of knowledge neurons from a set of action neurons and measures how they interact. This gap looks open.
- Not re-fetched this session (cite with care): Dai et al. 2022 "Knowledge Neurons" (arXiv 2104.08696); the critique by Niu et al. (ICLR 2024, arXiv 2405.02421); Gurnee et al. 2023 "Neurons in a Haystack" (arXiv 2305.01610), which the essay's appendix disputes; and Huang et al. 2023 on rigorously assessing neuron explanations (arXiv 2309.10312).

---

## Q4. How does the neuron view relate to manifold and geometry views (Manifold Steering, 2605.05115)?

### Takeaway
Mostly the two views are compatible at different scales and loci, with real tension over the residual stream. MLP neurons have a privileged basis because of the nonlinearity, and the essay's evidence lives there. Geometric and manifold accounts mostly describe the residual stream, which has no privileged basis. Goodfire publishes both lines of work.

### Cited Findings
- Goodfire published "Steering Along Manifolds to Control Neural Networks" (May 7, 2026), "Can SAEs Capture Neural Geometry?" (May 21), "A Geometric Calculator Inside a Neural Network" (May 14), "Meandering on Manifolds" (Jun 23) and "Uncovering Neural Geometry in Vision Models With Block-Sparse Featurizers" (Jul 7). The geometry program and the neuron essay come from the same lab. — [Goodfire research](https://www.goodfire.com/research)
- The essay does not discuss manifolds or geometry in its readable text. It says "I usually prefer specific mechanisms, like neurons, and specific pieces of data." Its closest geometric note is a "floppy ear dog" neuron covering "slightly different floppy ears at different angles", i.e. continuous variation inside one neuron. — [Essay](https://already-clean.goodfire.com)
- The essay concedes a residual-stream visualization problem (an "unread saliency knob") and that "attention mixes different kinds of mechanisms together". — [Essay](https://already-clean.goodfire.com)
- Nanda's main remaining crux is residual-stream superposition, i.e. more concepts than dimensions. — [implicator.ai](https://www.implicator.ai/goodfire-cammarata-ai-neurons-easier-to-read/)

### Inferences
- **Compatible reading.** A family of clean neurons can *tile* a manifold. The classic curve-detector family tiles orientation, and the essay's compressed layer merged them into coarser curve neurons. On this reading a manifold is a population-level description and each neuron is a local coordinate or "pole" on it. Manifold steering would act on the residual-stream geometry, which neuron families read from and write to.
- **Tension.** If knowledge-relevant quantities such as confidence, truth value or user stance live as non-axis-aligned curved structures in the residual stream, then neuron audits only see their *projections* through MLP read-in weights. The 2604.19117 finding of an orthogonal direction for opinions is that kind of residual-stream geometry claim.
- **Practical upshot for the project.** Use neurons for *mechanism localization and causal ablation* (what pushes the output). Use geometry for *state description* (where the knowledge representation sits and how it moves under pressure). The two can be checked against each other: does the knowledge-manifold position stay intact while say-not or holds-back neurons switch?

### Gaps
- I found no published direct comparison by Goodfire or others of neuron-level vs manifold-level accounts on the same LLM behavior.

---

## Q5. Concrete opportunities: if the claim holds, which cheap and novel knows-but-acts-against experiments open up? If it doesn't, what is the fallback?

### Takeaway
Yes. If the essay holds even roughly, individual neurons (and poles) can plausibly track knowledge and action separately, and the essay's own Qwen examples already show the pattern: concept or knowledge neurons upstream, and sparse "say not / hedge / holds back" override neurons downstream. Neuron attribution graphs have *no reconstruction-error nodes*. That makes the "bypassed" and "cancelled" hypotheses testable in a way transcoder or CLT graphs are not, because there an error node could hide the missing path. The fallback is head- and direction-level analysis plus PRISMS-style sparse neuron detectors, which do not depend on neurons being clean.

### Cited Findings (basis for the proposals)
- Neuron attribution graphs on Qwen3-32B "need no reconstruction loss", and four hub neurons reproduce Anthropic's diagnosis graph. — [Essay](https://already-clean.goodfire.com)
- Sparse override neurons exist and are causal: about 25 say-not neurons (confident-user case), about 19 hedge neurons, and one Kimi K3 holds-back neuron (P(return) from 0.83 to 0.98). — [Essay](https://already-clean.goodfire.com)
- Negative poles carry the answer in at least one case, Santa. — [Essay](https://already-clean.goodfire.com)
- Knowledge preserved while action flips, at the head level: ablating the wrong-signal heads changes deference, not accuracy. — [arXiv 2604.19117](https://arxiv.org/abs/2604.19117)
- Judgment and action are encoded at different token positions. — [arXiv 2507.11878](https://arxiv.org/abs/2507.11878)
- Sparse action-failure neurons are detectable before generation in agentic tool use. — [arXiv 2608.00218](https://arxiv.org/abs/2608.00218)
- Knowledge signals can be readable but not controllable, and distributed. — [arXiv 2607.00158](https://arxiv.org/abs/2607.00158); [arXiv 2604.19765](https://arxiv.org/abs/2604.19765)

### Inferences: proposed experiments (my proposals, not from sources)

**E0. Mini-replication of the audit, before building on it.** About 1–2 GPU-days. Sample about 300 MLP neuron poles from Qwen3-8B or 32B, stratified by layer, and run an agentic describe-then-verify loop with steering and ablation checks. Report the single-role share at two thresholds: the strong-activation band and the full range. This protects the project from the threshold-dependence critique and is itself a publishable early replication.

**E1. Neuron-level taxonomy of knows-but-acts-against.** The core experiment.
1. Build matched prompt pairs, using the same fact under a neutral query vs a pressured context: a confident user, a misleading system prompt, or an agentic tool context. Choose pairs where the model answers correctly when neutral and incorrectly under pressure.
2. Build neuron attribution graphs for both prompts.
3. Assign each failure to a category using these tests:
   - **Lost:** the knowledge neurons (the concept or answer neurons active in the neutral graph) drop below their natural activation band.
   - **Suppressed:** the knowledge neurons fire but the downstream say-X neurons are inhibited. Look for specific inhibitor neurons (say-not or holds-back types) whose ablation restores the answer.
   - **Cancelled:** the say-X neurons fire at baseline, but opposing poles or neurons contribute a roughly equal and opposite logit effect, so the net is near zero.
   - **Outcompeted:** say-X is intact at baseline magnitude, but say-Y has larger attribution, e.g. a user-name copy route.
   - **Bypassed:** the knowledge neurons fire at baseline, but their attribution to the output is near zero, because the path is cut in attention and the final answer is driven by a different route.

   The categories are mutually distinguishable with neuron activations plus attributions alone. Because nothing goes into an error node, "bypassed" is not an artifact of reconstruction error.

**E2. A catalog of override neurons across behaviors.** Collect say-not, hedge and holds-back neurons from sycophancy, refusal-over-knowledge, tool over- and under-calling (PRISMS tasks) and eval-awareness prompts. Test whether a *shared small family* of action-gating neurons mediates divergence across behaviors. This would be the neuron analogue of 2604.19117's shared sycophancy-lying circuit, and a novel test of whether that circuit has MLP-neuron counterparts.

**E3. A negative-pole sweep.** For each divergence case, rank neurons by |negative-pole attribution| vs positive. If suppressed knowledge sits preferentially in negative poles, as the Santa case hints, then existing description pipelines (OpenAI, Transluce top-activations) systematically miss it. This is cheap and novel.

**E4. Neuron graph vs transcoder or CLT graph on the same cases.** Measure how often a "bypassed" verdict in the transcoder graph comes from a large error node versus a real missing path in the neuron graph. This is a methodological contribution that holds regardless of who wins the superposition debate.

**E5. Neuron vs geometry cross-check.** Track whether a residual-stream knowledge direction or manifold position (probe or manifold coordinate) is unchanged under pressure while the override neurons switch. That would be the signature of suppression or outcompetition rather than loss.

### Inferences: fallback if neurons are not clean enough (e.g. E0 gives well under 90%, or knowledge neurons are distributed)
- Keep the *action* side at neuron level, since override and failure neurons appear sparse in PRISMS, safety neurons and the essay's examples. Move the *knowledge* side to probes, directions or heads, using 2604.19117 and 2507.11878 as templates.
- Treat "readable but not controllable" (2607.00158) as an operational signature of *bypass* or *redundancy* rather than as a failure.
- Use transcoders or CLTs, but report error-node mass explicitly for each category verdict.

### Inferences: risks
- Qwen3-32B neuron graphs are expensive. Start with Qwen3-8B, but note that the audit claim is specific to 32B.
- Agent-written neuron labels can be misleading. Labels are a poor guide to causal role, as in the Santa case where labeled neurons mattered less than negative-pole ones. Decide categories by causal ablation, not by labels.
- Attention routing is outside the neuron basis. Some "bypass" cases will be head-level and need path patching as in 2604.19117.

### Gaps
- No public neuron-graph tooling from Goodfire for Qwen3-32B was found. Whether the agent pipeline or graph code will be released with the companion paper is unknown.

### For other researchers (cross-cutting finds)
- **Mechanisms**
  - 2604.19117 has attention heads carrying "statement is wrong" under sycophancy. Ablating them changes deference but not accuracy, the circuit is shared with lying, and RLHF suppresses the behavior while the circuit persists.
  - 2507.11878 finds harmfulness and refusal as separate directions at different token positions.

  Both are strong "knowledge preserved, action diverges" results and are not in `papers_master.csv`.
- **Interventions**
  - EACL-2026 Safety Knowledge Neurons: ActCali gives above 97% ASR with about 0.3% of parameters changed, and SafeTuning is the defense.
  - "A Few Bad Neurons" (2601.18939) fine-tunes about 3% of neurons against sycophancy.
  - Safety Neurons (2406.14144): about 5% of neurons restore about 90% of safety performance.
  - 2607.00158 and 2604.19765 caution that hallucination neurons are readable but not controllable, and domain-specific.
- **Dynamics and training**
  - H-Neurons (2512.01797) originate in pretraining.
  - 2604.19117: RLHF lowers sycophancy about 10× while the head circuit persists or grows, which looks like suppression rather than removal.
  - The essay predicts that recursive generalization can be tested by watching neurons grow over training.
- **Labs survey.** In the last two months Goodfire also published:
  - "Models know when they're reward hacking — and we can catch them at scale" (Sep 17, 2026)
  - "Logits as a new monitor for evaluation awareness" (Jun 4)
  - "Verbalized Eval Awareness Inflates Measured Safety" (May 4)
  - "Reasoning Theater: Probing for Performative Chain-of-Thought" (Mar 12)

  All four are knows-vs-acts adjacent. — [Goodfire research](https://www.goodfire.com/research)
- **Datasets and tools.** Transluce has descriptions for all 458,752 Llama-3.1-8B-Instruct MLP neurons, both polarities. That is a ready-made label set for E3, the negative-pole study. — [Transluce](https://transluce.org/neuron-descriptions)
