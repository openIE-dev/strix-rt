# Compares crates/kev HF-json BPE (hf_tok, Laya/ModernBERT) with HF tokenizers on mixed-script text, incl. letter/mark/number edge cases.
import subprocess, sys
from tokenizers import Tokenizer
import os
S=["hello world 123 4567","naïve café résumé — déjà vu","Ⅻ ⅷ ½ ² ³ ① ٣٤٥ १२३","नमस्ते दुनिया क्षत्रिय","สวัสดีครับ ภาษาไทย","مرحبا بالعالم ١٢٣","שָׁלוֹם עוֹלָם","こんにちは世界 カタカナ ｶﾀｶﾅ","안녕하세요 세계","Привет мир ёЁ","ᚠᚢᚦ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𐌰𐌱","e\u0301 a\u0308 Z̤̈ ợ","emoji 👍🏽 👨‍👩‍👧 ✓","tab\tnew\\nline  spaces   x","ʰʲ ˈstrɛs ǅ ꙮ ᴀʙᴄ","ꯃꯤꯇꯩ ᮞᮥᮔ᮪ᮓ ᬩᬮᬶ","I'm don't YOU'LL we've","₁₂ ⁴ ႑႒ ௧௨ ꘠꘡","$1,234.56 + 7×8 = 56","ℕℤℚ ℵ ℓ ㉑ ⒈","two  spaces   three    four","x[MASK]y and  [MASK] z [CLS] [SEP]","email |||EMAIL_ADDRESS||| ip |||IP_ADDRESS|||","Incident log line 3: checkout latency p99 rose to 1011 ms; cards ending 1039.","  leading and trailing  ","multi\\n\\nnewlines \\n  indented","[unused3] tokens [PAD]<|endoftext|>"]
open("/tmp/tok_in.txt","w").write("\n".join(S)+"\n")
r=subprocess.run(["../../target/release/hf_tok",sys.argv[1],"/tmp/tok_in.txt"],env=dict(os.environ),capture_output=True,text=True,check=True).stdout.splitlines()
hf=Tokenizer.from_file(sys.argv[2]); bad=0
for s,l in zip(S,r):
    a=[int(x) for x in l.split()]; b=hf.encode(s.replace("\\n","\n"),add_special_tokens=False).ids
    if a!=b: bad+=1; print("MISMATCH",repr(s),a,b)
print(f"{len(S)-bad}/{len(S)} identical")
