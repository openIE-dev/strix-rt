import json, sys
for r in json.load(open(sys.argv[1])):
    print(",".join(map(str, r["ids"])) + "|" + ",".join(map(str, r["gen"])) + "|" + ",".join(f"{m:.4f}" for m in r["margin"]))
