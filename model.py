"""Reference implementation of a Transformer with verifiable answer reuse.

The weights are random and frozen. This program checks a structural property,
not language understanding. NumPy >= 1.26, Python >= 3.10, deterministic CPU.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np


def normalize(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def softmax(x):
    weights = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return weights / np.sum(weights, axis=-1, keepdims=True)


def route_vector(x):
    """A quantized key or query is part of the model definition."""
    return np.rint(127 * np.tanh(x)).astype(np.int8)


def scores(keys, query):
    # Both operands must be int32 before their products are accumulated.
    return keys @ query.astype(np.int32)


def readonly(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Page:
    tokens: tuple
    routing_key: np.ndarray
    keys: tuple
    values: tuple


@dataclass(frozen=True, eq=False)
class Bank:
    owner: object
    pages: Mapping[int, Page]
    ids: np.ndarray
    routing_keys: np.ndarray


@dataclass(frozen=True, eq=False)
class Edit:
    before: Bank
    after: Bank
    changed_ids: frozenset
    new_keys: np.ndarray


@dataclass(frozen=True, eq=False)
class Read:
    start: int
    layer: int
    query: np.ndarray
    selected: tuple
    threshold: int | None


@dataclass(frozen=True, eq=False)
class Trace:
    bank: Bank
    prompt: tuple
    tokens: tuple
    logits: tuple
    reads: tuple
    keys: tuple
    values: tuple


class Decoder:
    """Small causal decoder with independent page encoding and segment-level routing."""

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen", False):
            raise AttributeError("use a new Decoder for different weights or inference settings")
        object.__setattr__(self, name, value)

    def __init__(self, seed=37, width=24, router_width=16, layers=2,
                 segment=4, top_k=2, vocabulary=79):
        if min(width, router_width, layers, segment, top_k, vocabulary) < 1:
            raise ValueError("model dimensions must be positive")
        if 127 * 127 * router_width > np.iinfo(np.int32).max:
            raise ValueError("router dimension exceeds int32 dot-product capacity")
        self.width = width
        self.router_width = router_width
        self.layers = layers
        self.segment = segment
        self.top_k = top_k
        self.vocabulary = vocabulary
        rng = np.random.default_rng(seed)

        def project(inputs, outputs):
            return rng.normal(size=(inputs, outputs)) / np.sqrt(inputs)

        self.embedding = rng.normal(size=(vocabulary, width))
        self.enc_q, self.enc_k, self.enc_v = (project(width, width) for _ in range(3))
        self.page_route = project(width, router_width)
        self.output = project(width, vocabulary)
        self.blocks = []
        for _ in range(layers):
            block = {k: project(width, width) for k in
                     ("sq", "sk", "sv", "so", "cq", "ck", "cv", "co")}
            block["route"] = project(width, router_width)
            block["up"] = project(width, 2 * width)
            block["down"] = project(2 * width, width)
            self.blocks.append(block)
        self.scale = width ** -0.5
        self.rates = np.exp(-np.arange(width) / width * 7)
        for array in (self.embedding, self.enc_q, self.enc_k, self.enc_v,
                      self.page_route, self.output, self.rates):
            readonly(array)
        self.blocks = tuple(MappingProxyType({k: readonly(w) for k, w in b.items()})
                            for b in self.blocks)
        self._frozen = True

    def _tokens(self, values, empty=False):
        seq = tuple(values)
        if not seq and not empty:
            raise ValueError("token sequence must be nonempty")
        if any(type(t) not in (int, np.int32, np.int64) or not 0 <= t < self.vocabulary for t in seq):
            raise ValueError("token id outside vocabulary")
        return tuple(map(int, seq))

    def encode(self, tokens):
        seq = self._tokens(tokens)
        local = np.sin(np.arange(len(seq))[:, None] * self.rates)
        x = self.embedding[list(seq)] + local
        attention = softmax(((x @ self.enc_q) @ (x @ self.enc_k).T) * self.scale)
        state = normalize(x + attention @ (x @ self.enc_v))
        key = route_vector(np.mean(state, axis=0) @ self.page_route)
        return Page(seq, readonly(key), tuple(readonly(state @ b["ck"]) for b in self.blocks),
                    tuple(readonly(state @ b["cv"]) for b in self.blocks))

    def _bank(self, pages):
        ids = np.array(sorted(pages), dtype=np.int64)
        keys = self._key_matrix([pages[int(i)].routing_key for i in ids])
        return Bank(self, MappingProxyType(pages), readonly(ids), keys)

    def _key_matrix(self, vectors):
        return readonly(np.asarray(vectors, dtype=np.int32).reshape(-1, self.router_width))

    @staticmethod
    def _page_id(value):
        if not isinstance(value, (int, np.integer)) or not 0 <= value < 2**63:
            raise ValueError("page ids must be stable nonnegative int64s")
        return int(value)

    def bank(self, pages):
        return self._bank({self._page_id(i): self.encode(tokens) for i, tokens in pages.items()})

    def edit(self, before, changes):
        """Return the new version and its complete change set as one operation."""
        if before.owner is not self:
            raise ValueError("bank was built by a different model")
        pages = dict(before.pages)
        changed = set()
        fresh = []
        for page_id, tokens in changes.items():
            i = self._page_id(page_id)
            if tokens is None:
                if i not in pages:
                    raise ValueError(f"cannot delete missing page {i}")
                del pages[i]
            else:
                seq = self._tokens(tokens)
                if i in pages and seq == pages[i].tokens:
                    continue
                pages[i] = self.encode(seq)
                fresh.append(pages[i].routing_key)
            changed.add(i)
        return Edit(before, self._bank(pages), frozenset(changed), self._key_matrix(fresh))

    def _read(self, bank, query, start, layer):
        raw = scores(bank.routing_keys, query)
        # Python integers avoid negating a narrow integer at the cutoff.
        pos = sorted(range(len(raw)), key=lambda j: (-int(raw[j]), int(bank.ids[j])))[:self.top_k]
        selected = tuple(int(bank.ids[j]) for j in pos)
        threshold = int(raw[pos[-1]]) if len(pos) == self.top_k else None
        return Read(start, layer, readonly(query), selected, threshold)

    def _self_step(self, hidden, b, keys, values):
        normed = normalize(hidden)
        keys.append(readonly(normed @ b["sk"]))
        values.append(readonly(normed @ b["sv"]))
        weights = softmax((np.asarray(keys) @ (normed @ b["sq"])) * self.scale)
        return hidden + (weights @ np.asarray(values)) @ b["so"]

    @staticmethod
    def _ff(hidden, b):
        return hidden + np.tanh(normalize(hidden) @ b["up"]) @ b["down"]

    def _decode(self, bank, prompt, count, previous=None, start=0):
        if bank.owner is not self:
            raise ValueError("bank was built by a different model")
        if count < start or start % self.segment:
            raise ValueError("bad checkpoint")
        if previous is None:
            tokens, logits, reads = [], [], []
            keys = [[] for _ in self.blocks]
            values = [[] for _ in self.blocks]
            # Prompt states have no document dependency.
            for position, token in enumerate(prompt):
                x = self.embedding[token] + np.sin(position * self.rates)
                for layer, block in enumerate(self.blocks):
                    x = self._ff(self._self_step(x, block, keys[layer], values[layer]), block)
        else:
            n = len(prompt) + start
            tokens, logits = list(previous.tokens[:start]), list(previous.logits[:start])
            reads = [read for read in previous.reads if read.start < start]
            keys = [list(row[:n]) for row in previous.keys]
            values = [list(row[:n]) for row in previous.values]

        selections = {}
        for step in range(start, count):
            token = tokens[-1] if tokens else 0
            x = self.embedding[token] + np.sin((len(prompt) + step) * self.rates)
            for layer, b in enumerate(self.blocks):
                x = self._self_step(x, b, keys[layer], values[layer])
                if step % self.segment == 0:
                    query = route_vector(normalize(x) @ b["route"])
                    read = self._read(bank, query, step, layer)
                    reads.append(read)
                    selections[layer] = tuple(bank.pages[i] for i in read.selected)
                pages = selections[layer]
                if pages:
                    memory_k = np.concatenate([p.keys[layer] for p in pages], axis=0)
                    memory_v = np.concatenate([p.values[layer] for p in pages], axis=0)
                    cross = softmax((memory_k @ (normalize(x) @ b["cq"])) * self.scale)
                    x = x + (cross @ memory_v) @ b["co"]
                x = self._ff(x, b)
            prediction = normalize(x) @ self.output
            logits.append(readonly(prediction))
            tokens.append(int(np.argmax(prediction)))
        return Trace(bank, prompt, tuple(tokens), tuple(logits), tuple(reads),
                     tuple(map(tuple, keys)), tuple(map(tuple, values)))

    def generate(self, bank, prompt, count=16):
        if count < 1:
            raise ValueError("generate at least one token")
        return self._decode(bank, self._tokens(prompt), count)

    def certified_prefix(self, trace, edit):
        if trace.bank is not edit.before or trace.bank.owner is not self or edit.after.owner is not self:
            raise ValueError("trace and edit must belong to the same model and old document version")
        for read in trace.reads:
            if not edit.changed_ids.isdisjoint(read.selected):
                return read.start
            if len(edit.new_keys) and (read.threshold is None or
                                      np.any(scores(edit.new_keys, read.query) >= read.threshold)):
                return read.start
        return len(trace.tokens)

    def update(self, trace, edit):
        start = self.certified_prefix(trace, edit)
        if start == len(trace.tokens):
            return replace(trace, bank=edit.after), start
        return self._decode(edit.after, trace.prompt, len(trace.tokens), trace, start), start
