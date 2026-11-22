# RebuildLM design and reproduction guide

[日本語版](DESIGN.ja.md)

A maintainer changes one line in an API manual. Of the 1,000 answers generated from the old version, which can still be served verbatim? Even an answer that cited an unchanged page could be stale if someone added an exception on another page. RebuildLM records which pages its decoder selected and how highly a new page would have to score to enter that selection. After an edit, it can keep a verified prefix and regenerate the rest of the answer.

The runnable prototype is a two-layer NumPy Transformer with frozen random weights. It tests whether incremental updates give exactly the same result as decoding from scratch. The 1.5B architecture, training procedure, and GPU evaluation described below are proposed experiments.

## Reproduce the reference experiment

Clone the repository and run the checker on a CPU with Python 3.10 or newer.

```sh
git clone https://github.com/gtori123/RebuildLM.git
cd RebuildLM
python -m pip install -r requirements.txt
python check.py --output reproduction.json
```

The checker generates the weights and synthetic token sequences locally, prints the results, and writes them to `reproduction.json`. It checks the generated token IDs, the entire vocabulary logit vector at each position, and the routing trace against a full decode. The recorded run used Python 3.12.14 and NumPy 2.3.5. [Decoder](model.py) · [Checker](check.py) · [Recorded results](checked.json)

All 48 revisions start from the same 48-page document. There are 16 replacements, 16 insertions, and 16 deletions. Each revision is tested with the same 24 prompts, with 16 generated tokens per answer. Each layer selects up to two pages at the start of every four-token segment. The model and data seeds are 37 and 12302.

| Measurement | Recorded result |
| --- | ---: |
| Revision–prompt pairs | 1,152 |
| Whole answers reused | 845 |
| Answers with only a prefix reused | 220 |
| Answers regenerated from the first segment | 87 |
| Token positions whose complete vocabulary logits were checked | 18,432 |
| Incremental answers with any token or logit mismatch | 0 |

A baseline reuses an answer whenever all previously selected pages are unchanged. It reused 959 whole answers and returned a different token sequence from a fresh decode in 87 of them. In one recorded case, page 9 first entered the selection in the second segment after a replacement. The first four tokens stayed the same, but the fifth changed from token ID 9 to token ID 5. The baseline's 87 errors and RebuildLM's 87 restarts count different events. The checker also covers no-op edits, ties at the cutoff, simultaneous deletion and insertion, empty and below-capacity page sets, successive edits, restoring old content, and applying an edit to the wrong version. [Results by edit type and example token sequences](checked.json)

## Research hypothesis

An answer may draw on several parts of a manual while relying on only a few pages at any one point. If the decoder can select pages again as it writes, it may be able to answer questions that span pages and still reuse much of its computation after a local edit. Two effects could defeat this hypothesis. A four-page limit could hide a qualification that changes the correct answer. A longer answer also makes more reads, any one of which an edit can invalidate. Checking and storing those reads could cost more than the generation they save.

The intended demonstration uses two versions of a library manual, such as v1.8 and v1.9, and a collection of FAQs, migration guides, and worked examples. A reader should be able to inspect the source diff beside the old and updated answers, see where generation resumed, and compare each update with an independent full decode. The demonstration would report elapsed time until all answers are ready, GPU usage, and storage under the same conditions, and let readers substitute their own document versions and questions.

## Proposed 1.5B architecture

| Component | First configuration to test |
| --- | --- |
| Backbone | Start with Qwen2.5-1.5B-Instruct. Insert document cross-attention after self-attention in layers 7, 14, 21, and 28 of its 28-layer decoder, counting from one. |
| Inputs | Keep versioned documents separate from the question and conversation. Prompt processing is independent of document content. Only the designated document-reading layers can combine the two inputs. |
| Page encoder | Give each section or function a stable ID. Split long sections into pages of roughly 256 tokens. Encode each page independently with the bottom four backbone layers, using shared or copied weights. Produce one routing key per page and per-token keys and values for cross-attention. |
| Routing | At the first position of each eight-token answer segment, each document-reading layer selects up to four pages. Keep that layer's selection for the segment. A later layer can use information read by an earlier layer to choose its own pages. |
| Document read | Apply cross-attention over tokens in the selected pages and return the result through a gated residual connection. The decoder receives neither a global document summary nor an attention normalization term computed over all pages. |
| Ranking | Quantize routing queries and page keys to 64-dimensional signed 8-bit vectors. Compute dot products in 32-bit integer arithmetic. Rank by descending score, then ascending stable ID. Audit runs use exact top-four selection. |
| Stored state | For each read, keep its query, ordered page IDs, and cutoff score. Associate the trace with the model, input, decoding settings, and document revision. Include the cost of retaining or reconstructing the self-attention key/value (KV) state needed to resume generation. |

Positions within a page are local. When a heading or a page's order affects its meaning, encode that metadata as part of its content. Renumbering then counts as an edit to every affected page. Represent splits, deletions, and insertions with the complete set of removed, added, and rewritten IDs. For widespread revisions, compare the cost of checking old reads with a full rebuild.

## When can a cached read be reused?

Let `q` be the stored routing query, `S` its ordered selection, and `τ` the score of the lowest-ranked selected page when the read selected exactly `k` pages. Accept the old read if both conditions hold.

1. Every page in `S` still exists with the same content, token representations, and positions.
2. Every added or rewritten page `p` satisfies `q · key(p) < τ`. A tie triggers recomputation even when the ID tie-breaker would preserve the result. If fewer than `k` pages were available, adding a candidate triggers recomputation.

Deleting an unselected page cannot move it into the selection. Unchanged candidates retain their scores because each page is encoded independently. If the two checks pass, the same pages reach cross-attention in the same order with identical keys and values.

The first routing query depends only on the prompt and the initial decoder state. Once a read passes, the decoder performs the same computations up to the next read, so its saved query is still valid. Apply that argument in generation order and, within a segment, in layer order. On the first failure, restore the self-attention state from the start of that segment and regenerate through the end. If every read passes, return the saved answer.

With the same weights, prompt, greedy decoder, and deterministic numerical settings, a verified prefix has exactly the same vocabulary logits and tokens as a fresh decode on the **revised** documents. Regenerating the remaining suffix then gives the same complete answer. Factual accuracy needs its own evaluation. Changes to weights, instructions, or external tool results require a new cache version. On a GPU, batch shapes can change rounding. The proposed evaluation would measure a deterministic audit mode and a throughput-oriented mode separately.

## Training plan

1. Train the page encoder, document-reading layers, and gates on questions and answers derived from the same documents. Use a dense document-reading model's next-token distributions as a teacher. If necessary, adapt some backbone weights with LoRA.
2. Collect pairs of public specifications or manuals across real revisions. Label answers and supporting pages in both versions for questions whose answers change and questions whose answers stay the same. Split development and evaluation data by project and time, and check reference answers by hand. Include exceptions added to previously unread pages, questions requiring multiple pages, and broad revisions.
3. Combine a next-token loss with a ranking loss that puts evidence pages above the competitor at the top-`k` boundary. For unrelated edits, test an auxiliary loss that keeps the new page below the cutoff by a margin. Keep hard top-`k` selection in the forward pass, supervise the router separately, and approximate gradients through quantization. Compare against a fixed router, training without the margin loss, and retrieval only at the start of an answer.
4. Report initial accuracy and post-edit accuracy separately for questions with changed and unchanged answers. Give changed-answer questions enough weight that a model that always serves the old answer cannot pass.

## Evaluation criteria for the trained model

Use real revision histories from multiple projects, with roughly 16,000–32,000 tokens of public technical documentation and up to 1,000 questions or explanations per version. Report fact lookup, questions spanning a rule and its exceptions, and long answers such as migration instructions separately. Report results both at the observed edit frequencies and with equal weight per edit type.

| Baseline | What it tests |
| --- | --- |
| Decode every answer from scratch with the same RebuildLM | Exact agreement and the time saved by incremental updates. |
| Give the full document to a comparably sized backbone, using its standard cache where available | The answer-quality cost of restricting access to a subset of pages. |
| Rerun exact top-`k` retrieval for each question and return its cached answer when the pages, order, and contents match | The strongest simple alternative. This form of retrieval-augmented generation (RAG) can also agree with its own full decode under deterministic settings. |
| Encode pages independently and read them all, or retrieve only once per answer | The contributions of independent encoding, repeated selection, and trace checking. |

Use the same GPU and batching budget for all comparisons. Post-edit accuracy should be within **two percentage points** of a strong full-context baseline, including on questions whose answers actually change. For each revision, time how long it takes until all updated answers are ready. The target is a **p95 update time** at most one fifth of the full RebuildLM rebuild p95 and lower than the p95 of exact re-retrieval RAG at comparable accuracy. Report confidence intervals by project. Include page re-encoding, scoring changed candidates against saved queries, KV reads and writes, and suffix generation in the timing.

Checking cost scales with the number of stored reads, the number of added or rewritten pages, and the routing dimension. With a 64-byte query, four 8-byte page IDs, and a 4-byte cutoff, one read takes about 100 bytes before headers and indexing. At 128 tokens per answer, eight tokens per segment, and four reading layers, 1,000 answers produce about **6.4 MB** of raw routing traces. KV storage is an additional cost. Compare saving KV state for every answer, saving it only for frequently used answers, and recomputing accepted prefixes when needed.

## Related work

- **CEPE** independently encodes context chunks for an existing decoder. [Long-Context Language Modeling with Parallel Context Encoding](https://aclanthology.org/2024.acl-long.142/)
- **Unlimiformer** lets decoder layers retrieve a top-`k` subset of encoder states for cross-attention during generation. [Unlimiformer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6f9806a5adc72b5b834b27e4c7c0df9b-Abstract-Conference.html)
- **PatchKV** studies which regions of a KV cache to repair after an edit inside the input. [PatchKV](https://arxiv.org/abs/2609.26219)
- **GroundedCache** uses signals such as question similarity, evidence, and document versions to decide whether to return an earlier answer. [Grounded Cache Routing for Retrieval-Augmented Generation](https://arxiv.org/abs/2605.27494)
- **Self-adjusting computation** provides the precedent for propagating a change while preserving agreement with a from-scratch run. [An Experimental Analysis of Self-Adjusting Computation](https://www.cs.cmu.edu/~guyb/papers/ABBHT09.pdf)
- **FreshStack** has been used to study how changing documents make retrieval benchmarks go stale. The proposed RebuildLM evaluation would require new answers and change labels for both revisions. [Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks](https://arxiv.org/abs/2603.04532)
- The proposed backbone's configuration is documented in the [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

The research question these comparisons address is whether restricting document access by layer and generation segment makes prior answers cheap enough to verify on real revision histories. Exact re-retrieval RAG is a particularly important comparison because it can already reuse answers after edits. RebuildLM would have to earn the cost of its finer-grained reads by preserving answer quality and reducing total update time.
