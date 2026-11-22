"""Reproduce the structural experiment with random frozen weights and synthetic token ids."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np

from model import Decoder


def equal_logits(left, right):
    return len(left.logits) == len(right.logits) and all(
        np.array_equal(a, b) for a, b in zip(left.logits, right.logits))


def require_match(actual, expected):
    if actual.tokens != expected.tokens or not equal_logits(actual, expected):
        raise AssertionError("incremental and full generation differ")
    for a, b in zip(actual.reads, expected.reads, strict=True):
        if ((a.start, a.layer, a.selected, a.threshold) != (b.start, b.layer, b.selected, b.threshold)
                or not np.array_equal(a.query, b.query)):
            raise AssertionError("read traces differ")


def edge_cases(model, bank, traces):
    t = traces[0]
    n = len(t.tokens)
    first_read = t.reads[0]
    # A no-op must survive in full, including against a new immutable snapshot.
    no_change = model.edit(bank, {1: bank.pages[1].tokens})
    out, kept = model.update(t, no_change)
    assert kept == n
    require_match(out, t)
    # Ties with the old Kth page conservatively require recomputation.
    tied = model.edit(bank, {600: bank.pages[first_read.selected[-1]].tokens})
    out, kept = model.update(t, tied)
    assert kept == 0
    require_match(out, model.generate(tied.after, t.prompt, n))
    # Deletion of a selected page and an insertion are a single transaction.
    both = model.edit(bank, {first_read.selected[0]: None, 700: (5, 7, 9, 11)})
    out, kept = model.update(t, both)
    assert kept == 0
    require_match(out, model.generate(both.after, t.prompt, n))
    # Reuse the updated trace for the next change, then restore the base version's content.
    restored = model.edit(both.after, {first_read.selected[0]: bank.pages[first_read.selected[0]].tokens,
                                       700: None})
    out2, _ = model.update(out, restored)
    require_match(out2, t)
    try:
        model.update(t, restored)
    except ValueError:
        pass
    else:
        raise AssertionError("a delta applied to the wrong revision was accepted")
    # Empty and below-K banks are allowed. Adding a new page below capacity always reranks.
    empty = model.bank({})
    old = model.generate(empty, (5, 2), 7)
    small = model.edit(empty, {1: (5, 9, 8)})
    out, kept = model.update(old, small)
    assert kept == 0
    require_match(out, model.generate(small.after, old.prompt, 7))
    cleared = model.edit(small.after, {1: None})
    out, _ = model.update(out, cleared)
    require_match(out, old)


def experiment():
    seed = 12302
    rng = np.random.default_rng(seed)
    model = Decoder()
    n_pages, n_queries, n_revisions, length = 48, 24, 48, 16

    def token_ids(n):
        return tuple(map(int, rng.integers(1, model.vocabulary, size=n)))

    bank = model.bank({i: token_ids(8) for i in range(n_pages)})
    traces = [model.generate(bank, token_ids(6), length) for _ in range(n_queries)]
    edge_cases(model, bank, traces)
    total = Counter()
    by_operation = {k: Counter() for k in ("replace", "insert", "delete")}
    example = None
    for version in range(n_revisions):
        op = ("replace", "insert", "delete")[version % 3]
        i = n_pages + version if op == "insert" else int(rng.integers(n_pages))
        edit = model.edit(bank, {i: None if op == "delete" else token_ids(8)})
        for q, old in enumerate(traces):
            fresh = model.generate(edit.after, old.prompt, length)
            updated, kept = model.update(old, edit)
            require_match(updated, fresh)
            old_ids = set().union(*(set(read.selected) for read in old.reads))
            naive_reuses = old_ids.isdisjoint(edit.changed_ids)
            stale_logits = naive_reuses and not equal_logits(old, fresh)
            stale_text = naive_reuses and old.tokens != fresh.tokens
            counts = dict(answers=1, checked_logits_vectors=length, retained_prefix_tokens=kept,
                          reused_entire_answer=int(kept == length), reused_partial_answer=int(0 < kept < length),
                          old_ids_only_reused=int(naive_reuses), old_ids_only_wrong_logits=int(stale_logits),
                          old_ids_only_wrong_tokens=int(stale_text), total_tokens=length)
            total.update(counts)
            by_operation[op].update(counts)
            if example is None and stale_text:
                example = dict(revision=version, operation=op, query_index=q, edited_page=i,
                               old_selected_ids=sorted(old_ids), prompt_token_ids=old.prompt,
                               newly_accepted_page_tokens=edit.after.pages[i].tokens, retained_prefix=kept,
                               old_output=old.tokens, full_rebuild_output=fresh.tokens)
    assert total["old_ids_only_wrong_tokens"] > 0, "test did not exercise a new relevant page"
    assert total["reused_partial_answer"] > 0
    return dict(scope="Random frozen decoder and synthetic tokens. No natural-language quality or speed measured.",
                python=platform.python_version(), numpy=np.__version__, weights_seed=37, data_seed=seed,
                dimensions=dict(width=model.width, router_width=model.router_width, layers=model.layers,
                                top_k=model.top_k, segment=model.segment),
                base_pages=n_pages, independent_revisions=n_revisions, fixed_prompts=n_queries,
                tokens_per_answer=length, incremental_vs_full_mismatches=0,
                total=dict(total), per_operation={k: dict(v) for k, v in by_operation.items()},
                example_missed_by_old_ids_only=example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment()
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
