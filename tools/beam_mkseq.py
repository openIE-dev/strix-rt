import re
# builds logs/beam/seq.tsv: name, full id sequence (prompt + baseline greedy tokens), prompt_len
out = open("logs/beam/seq.tsv", "w")
for l in open("logs/beam/prompts.tsv"):
    n, ids = l.rstrip("\n").split("\t"); ids = [int(x) for x in ids.split(",")]
    gen = [int(m.group(1)) for m in (re.match(r"pos \d+ top5 (\d+):", x) for x in open(f"logs/beam/prof/{n}.log")) if m]
    seq = ids + gen[:-1]
    out.write(n + "\t" + ",".join(map(str, seq)) + "\t" + str(len(ids)) + "\n")
