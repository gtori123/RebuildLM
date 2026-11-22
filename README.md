# RebuildLM

Which generated answers can you keep after their source documents change?

Imagine adding a mobile-only exception to an API manual. A previous answer may now be out of date even when every page it read is unchanged. RebuildLM records enough about each document lookup to detect when an edited or newly added page would enter the model's attention. It can then keep the verified prefix of an answer and regenerate the rest.

**Research prototype.** The runnable model is a two-layer NumPy Transformer with frozen random weights. The experiment tests exact reuse on synthetic token sequences. Natural-language answer quality and end-to-end speed on a trained 1.5B model are open experiments.

[日本語の設計書・実行手順](DESIGN.ja.md) · [Reference decoder](model.py) · [Recorded results](checked.json)

## Reproduce the experiment

From the repository root, run the following commands with Python 3.10 or newer.

```sh
python -m pip install -r requirements.txt
python check.py --output reproduction.json
```

The checker generates its data on the CPU and compares each incremental run with a fresh decode of the edited documents. It raises an error if the output tokens, vocabulary logits, or routing trace differ. The recorded run used Python 3.12.14 and NumPy 2.3.5.

| Result on 48 document edits × 24 prompts | Recorded value |
| --- | ---: |
| Answer comparisons | 1,152 |
| Whole answers reused | 845 |
| Additional answers with a reusable prefix | 220 |
| Token positions checked against fresh logits | 18,432 |
| Token or logit mismatches after incremental updating | 0 |
| Incorrect output sequences served by a baseline checking only previously selected page IDs | 87 of 959 reused answers |

One example makes the failure concrete. Page 9 was absent from every document lookup during the old answer. After a replacement, it entered the selection at the second four-token segment. The first four tokens survived, and the fifth changed from token ID 9 to token ID 5. Both sequences and the old page selections are in [the recorded example](checked.json).

## How reuse works

1. Encode each page independently, with stable page IDs and page-local positions.
2. At the start of each answer segment, the decoder chooses its top pages separately at each document-reading layer. Each read records its quantized query, selected IDs, and the cutoff score.
3. After a document edit, check reads in generation order. A read can be reused if its selected pages retain their content and every added or rewritten candidate scores strictly below the old cutoff. Ties trigger recomputation.
4. When a check fails, resume from the start of that segment. If every check passes, return the cached answer.

The equivalence claim applies to a fresh run of the **same** model, prompt, and deterministic inference settings. Factual correctness requires a separate evaluation. The larger-model design and planned comparisons against full-context decoding and an exact re-retrieval RAG baseline are in [the design document](DESIGN.ja.md).

## License

[MIT](LICENSE).
