import json,subprocess,sys,re
M="$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
P=json.load(open("logs/glimmer/prompts.json"))+json.load(open("logs/glimmer/prompts_long.json"))
P+=["Hello  world 123, café","I\x27m sure they\x27ll say it\x27s DON\x27T STOP, we\x27VE won","HTTPServerError camelCaseWord ÀÉÎ straße ǅungla","日本語のテキストと中文，한국어 тест Привет мир","x = [1,2,3]\n\n\n    y = {\"a\": 1}  \n\t\ttab\r\nCRLF line\r\n","12345678901234 3.14159 -42 +7 1,000,000","path/to/file.rs://a//b ... --- !!! ??? \n/ \n\n/x","  leading and trailing   ","<|image_start|>hi<|image_end|>\nafter <|begin_of_text|>x","emoji 😀👍🏽 and zero\u200bwidth, combining e\u0301 é","ALLCAPS lowercase MiXeD UPPERlower lowerUPPER","\n","a","   ","\u00a0nbsp\u2003em  space","don\x27t\x27\x27t \x27S \x27re \x27 \x27"]
bad=0
for p in P:
    ref=subprocess.run([ "~/llama-latest/build-vulkan/bin/llama-tokenize","-m",M,"--ids","-p",p],capture_output=True,text=True).stdout.strip().splitlines()[-1]
    ref=ref.strip("[]").replace(" ","")
    ours=subprocess.run(["target-glm/release/glm_tok",M,p],capture_output=True,text=True).stdout.strip()
    ok= ref==ours
    if not ok: bad+=1; print("MISMATCH",repr(p),"\n ref ",ref,"\n ours",ours)
print(len(P),"prompts,",bad,"mismatches")
