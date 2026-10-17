// reference dumper: mcvdump <model> <mmproj> <image> <prompt-file> <outdir> <n_gen> [vocab_only]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 1; }
    std::string outdir = argv[5]; int n_gen = atoi(argv[6]);
    FILE * pf = fopen(argv[4], "rb"); std::string prompt; char b[4096]; size_t r;
    while ((r = fread(b, 1, sizeof b, pf)) > 0) prompt.append(b, r); fclose(pf);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    auto cp = mtmd_context_params_default(); cp.print_timings = true; cp.warmup = false; if (getenv("MTMD_CPU")) cp.use_gpu = false;
    mtmd_context * mctx = mtmd_init_from_file(argv[2], model, cp);
    bool noimg = std::string(argv[3]) == "-";
    mtmd_helper_bitmap_wrapper bw = {};
    if (!noimg) bw = mtmd_helper_bitmap_init_from_file(mctx, argv[3], false, mtmd_helper_init_opt_default());
    mtmd_input_text txt = { prompt.c_str(), prompt.size(), true, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bms[1] = { bw.bitmap };
    if (mtmd_tokenize(mctx, chunks, &txt, bms, noimg ? 0 : 1) != 0) { fprintf(stderr, "tokenize fail\n"); return 1; }
    FILE * tf = fopen((outdir + "/chunks.txt").c_str(), "w");
    int n_embd = llama_model_n_embd_inp(model);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n; auto * t = mtmd_input_chunk_get_tokens_text(c, &n);
            fprintf(tf, "TEXT %zu:", n); for (size_t j = 0; j < n; j++) fprintf(tf, " %d", t[j]); fprintf(tf, "\n");
        } else {
            size_t nt = mtmd_input_chunk_get_n_tokens(c);
            auto * it = mtmd_input_chunk_get_tokens_image(c);
            fprintf(tf, "IMAGE n_tokens=%zu n_pos=%d\n", nt, (int) mtmd_input_chunk_get_n_pos(c));
            // encode twice, time the second
            mtmd_encode_chunk(mctx, c);
            double t0 = now(); mtmd_encode_chunk(mctx, c); double t1 = now();
            fprintf(tf, "ENCODE_S %.4f\n", t1 - t0);
            float * e = mtmd_get_output_embd(mctx);
            FILE * ef = fopen((outdir + "/img_embd_" + std::to_string(i) + ".f32").c_str(), "wb");
            fwrite(e, sizeof(float), nt * (size_t) n_embd, ef); fclose(ef);
            FILE * qf = fopen((outdir + "/img_pos_" + std::to_string(i) + ".txt").c_str(), "w");
            for (size_t j = 0; j < nt; j++) { auto p = mtmd_image_tokens_get_decoder_pos(it, 0, j); fprintf(qf, "%u %u %u\n", p.t, p.y, p.x); }
            fclose(qf);
        }
    }
    fclose(tf);
    if (n_gen <= 0) return 0;
    auto ccp = llama_context_default_params(); ccp.n_ctx = 8192; ccp.n_batch = 2048; ccp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, ccp);
    llama_pos np = 0;
    double t0 = now();
    if (mtmd_helper_eval_chunks(mctx, ctx, chunks, 0, 0, 512, true, &np)) { fprintf(stderr, "eval fail\n"); return 1; }
    double t1 = now();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<int> out; std::string text;
    double t2 = now();
    for (int k = 0; k < n_gen; k++) {
        float * lg = llama_get_logits_ith(ctx, -1); int nv = llama_vocab_n_tokens(vocab);
        int best = 0; for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
        out.push_back(best);
        char pc[256]; int l = llama_token_to_piece(vocab, best, pc, sizeof pc, 0, true); if (l > 0) text.append(pc, l);
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_token tk = best; llama_batch bt = llama_batch_get_one(&tk, 1);
        if (llama_decode(ctx, bt)) { fprintf(stderr, "decode fail\n"); return 1; }
    }
    double t3 = now();
    FILE * of = fopen((outdir + "/gen.txt").c_str(), "w");
    fprintf(of, "N_POS_PROMPT %d\nPREFILL_EVAL_S %.4f\nDECODE_S %.4f N %zu\nIDS", (int) np, t1 - t0, t3 - t2, out.size());
    for (int x : out) fprintf(of, " %d", x);
    fprintf(of, "\nTEXT %s\n", text.c_str()); fclose(of);
    printf("%s\n", text.c_str());
    return 0;
}
