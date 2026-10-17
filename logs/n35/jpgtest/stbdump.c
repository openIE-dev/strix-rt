#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
// stbdump <in.jpg> <out.rgb> : stbi_load_from_memory(req_comp=3) as llama.cpp mtmd does; out = i32 w, i32 h, RGB8
int main(int c, char **v) { FILE *f = fopen(v[1], "rb"); if (!f) return 2; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); unsigned char *b = malloc(n); fread(b, 1, n, f); fclose(f);
  int w, h, nc; unsigned char *d = stbi_load_from_memory(b, n, &w, &h, &nc, 3); if (!d) { printf("FAIL %s %s\n", v[1], stbi_failure_reason()); return 1; }
  FILE *o = fopen(v[2], "wb"); fwrite(&w, 4, 1, o); fwrite(&h, 4, 1, o); fwrite(d, 1, (size_t)w*h*3, o); fclose(o); printf("OK %s %dx%d comp %d\n", v[1], w, h, nc); return 0; }
