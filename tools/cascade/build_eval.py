#!/usr/bin/env python3
"""Build the cascade eval set: 4 public datasets (small samples via the HF datasets-server rows API, stdlib only).
Kev's model card lists these among its training sources (train splits); we use held-out test/validation rows.
Output tools/cascade/eval.jsonl: {id, ds, state, qtype, instructions, options{key:desc}, gold(key)}; seeded, 250 per dataset."""
import json, random, urllib.request, urllib.parse
N = 250
def rows(ds, cfg, split, offs, n=100):
    out = []
    for o in offs:
        u = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(dict(dataset=ds, config=cfg, split=split, offset=o, length=n))
        out += [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=60))["rows"]]
    return out
rnd = random.Random(1234); items = []
def add(ds, rs, f):
    rnd.shuffle(rs)
    for i, r in enumerate(rs[:N]):
        it = f(r); it["id"] = f"{ds}-{i}"; it["ds"] = ds; items.append(it)
add("ag_news", rows("fancyzhx/ag_news", "default", "test", [0, 2500, 5000]), lambda r: dict(
    state=r["text"], qtype="choice", instructions="Which news topic does this article belong to?",
    options={"world": "international politics, conflicts, governments", "sports": "sports events and athletes", "business": "companies, markets, economy", "scitech": "science and technology"},
    gold=["world", "sports", "business", "scitech"][r["label"]]))
add("sst5", rows("SetFit/sst5", "default", "test", [0, 900, 1800]), lambda r: dict(
    state=r["text"], qtype="choice", instructions="What is the sentiment of this movie review?",
    options={"very_negative": "strongly negative", "negative": "negative", "neutral": "neutral or mixed", "positive": "positive", "very_positive": "strongly positive"},
    gold=r["label_text"].replace(" ", "_")))
NLI = ["entailment", "neutral", "contradiction"]
add("mnli", rows("nyu-mll/multi_nli", "default", "validation_matched", [0, 3300, 6600]), lambda r: dict(
    state="Premise: " + r["premise"] + "\nHypothesis: " + r["hypothesis"], qtype="choice", instructions="What is the relationship of the hypothesis to the premise?",
    options={"entailment": "the premise implies the hypothesis", "neutral": "the premise neither implies nor contradicts it", "contradiction": "the premise contradicts the hypothesis"},
    gold=NLI[r["label"]]))
add("boolq", rows("google/boolq", "default", "validation", [0, 1100, 2200]), lambda r: dict(
    state="Question: " + r["question"] + "?\nPassage: " + r["passage"], qtype="noul", instructions="Does the passage show that the answer to the question is yes?",
    options={"no": "", "yes": ""}, gold="yes" if r["answer"] else "no"))
with open("tools/cascade/eval.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
print(len(items), {d: sum(1 for i in items if i["ds"] == d) for d in set(i["ds"] for i in items)})
