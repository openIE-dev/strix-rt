// Dequantize rows of a GGUF tensor with a llama.cpp fork's own ggml (to_float type trait).
// build: cc -O2 fork_dequant.c -I$FORK/ggml/include -L$FORK/build-cpu/bin -lggml-base -o fork_dequant_x
// usage: fork_dequant_x <gguf> <tensor> <row0> <nrows> <out.f32>
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    struct ggml_context *meta = NULL;
    struct gguf_init_params ip = { .no_alloc = true, .ctx = &meta };
    struct gguf_context *g = gguf_init_from_file(argv[1], ip);
    if (!g) { fprintf(stderr, "gguf open failed\n"); return 1; }
    int64_t ti = gguf_find_tensor(g, argv[2]);
    if (ti < 0) { fprintf(stderr, "no tensor\n"); return 1; }
    struct ggml_tensor *t = ggml_get_tensor(meta, argv[2]);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
    long r0 = atol(argv[3]), nr = atol(argv[4]);
    size_t rb = ggml_row_size(t->type, t->ne[0]);
    const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
    if (!tr->to_float) { fprintf(stderr, "no to_float for type %d\n", t->type); return 1; }
    FILE *f = fopen(argv[1], "rb"); fseeko(f, off + r0 * rb, SEEK_SET);
    char *buf = malloc(rb * nr); if (fread(buf, rb, nr, f) != (size_t)nr) { fprintf(stderr, "short read\n"); return 1; }
    float *y = malloc(sizeof(float) * t->ne[0] * nr);
    for (long r = 0; r < nr; r++) tr->to_float(buf + r * rb, y + r * t->ne[0], t->ne[0]);
    FILE *o = fopen(argv[5], "wb"); fwrite(y, sizeof(float), t->ne[0] * nr, o); fclose(o);
    printf("type=%d (%s) ne0=%lld rowbytes=%zu rows=%ld..%ld\n", t->type, ggml_type_name(t->type), (long long)t->ne[0], rb, r0, r0 + nr);
    return 0;
}
