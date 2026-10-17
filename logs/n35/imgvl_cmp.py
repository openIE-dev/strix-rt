import subprocess,re,sys,os
# imgvl_cmp.py <name>: compare our greedy ids (imgref/<name>.vl.out) with llama.cpp mtmd-cli answer (imgref/<name>.ans) tokenised by llama-tokenize.
# The mtmd-cli answer file starts with a newline and ends with the EOS-adjacent text; a final ".\n\n" token vs "." + EOS is reported as equivalent.
T=os.path.expanduser("~/llama-n35/build-vulkan/bin/llama-tokenize"); M=os.path.expanduser("$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf")
b=sys.argv[1]
ref=eval(subprocess.check_output([T,"-m",M,"--no-bos","--ids","--log-disable","-f","imgref/%s.ans"%b]).decode())[1:]
ours=eval(re.search(r"^ids: (\[.*\])",open("imgref/%s.vl.out"%b).read(),re.M).group(1))
k=0
while k<min(len(ref),len(ours)) and ref[k]==ours[k]: k+=1
tail_ok = (k==len(ref)-1 and len(ours)>k+1 and ours[k+1]==11)  # ours emits "." then <|im_end|>(11) where llama.cpp has the merged ".\n\n" token
print("%s: identical prefix %d of ref %d ids%s"%(b,k,len(ref)," (final token only differs by merge '.' + EOS vs '.\\n\\n': text identical)" if tail_ok else ("" if k==len(ref) else "; first diff ref %s ours %s"%(ref[k:k+3],ours[k:k+3]))))
