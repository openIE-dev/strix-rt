// mtmd_dump TEXT_GGUF MMPROJ IMAGE OUT.f32 [gpu]: encode one image with llama.cpp mtmd, write [n_tok][n_embd] f32 (+ header "n_tok n_embd" to stdout)
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
int main(int argc, char ** argv) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = argc > 5; cp.warmup = false; cp.n_threads = 16; cp.print_timings = true;
    mtmd_context * c = mtmd_init_from_file(argv[2], m, cp);
    if (!c) { fprintf(stderr, "mtmd init failed\n"); return 1; }
    auto w = mtmd_helper_bitmap_init_from_file(c, argv[3], false, mtmd_helper_init_opt_default());
    if (!w.bitmap) { fprintf(stderr, "image load failed\n"); return 1; }
    mtmd_input_text t; std::string s = std::string(mtmd_default_marker()); s += "\nThis image shows"; t.text = s.c_str(); t.add_special = true; t.parse_special = true;
    mtmd_input_chunks * ch = mtmd_input_chunks_init();
    const mtmd_bitmap * bm[1] = { w.bitmap };
    int r = mtmd_tokenize(c, ch, &t, bm, 1);
    if (r) { fprintf(stderr, "tokenize %d\n", r); return 1; }
    for (size_t i = 0; i < mtmd_input_chunks_size(ch); i++) {
        const mtmd_input_chunk * k = mtmd_input_chunks_get(ch, i);
        if (mtmd_input_chunk_get_type(k) == MTMD_INPUT_CHUNK_TYPE_TEXT) { size_t nt; const llama_token * tk = mtmd_input_chunk_get_tokens_text(k, &nt); printf("TEXT"); for (size_t j = 0; j < nt; j++) printf(" %d", tk[j]); printf("\n"); continue; }
        printf("IMAGE\n");
        if (mtmd_encode_chunk(c, k)) { fprintf(stderr, "encode failed\n"); return 1; }
        size_t n = mtmd_input_chunk_get_n_tokens(k); int ne = 6656;
        FILE * f = fopen(argv[4], "wb"); fwrite(mtmd_get_output_embd(c), sizeof(float), n * ne, f); fclose(f);
        printf("n_tok %zu n_embd %d\n", n, ne);
    }
    return 0;
}
