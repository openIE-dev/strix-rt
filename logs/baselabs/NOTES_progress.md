## 2026-10-30 progress (UTC)
- 15:53 BUG FOUND: prefill::run redirected "attn" -> causal attnfa (PF_FA=1), so bidir kernel never ran (joey parity v1 = causal: 19/20 argmax, max dlogit 1.02). Fix: pf.k "attn_nofa" marker skips redirect (prefill/src/lib.rs:233, kev lib.rs base_labs). Rebuilt 15:53.
- Queued under gpu_lock: elephant_parity+joey25_parity (old/new binary at exec), joey_causal_parity (BL_CAUSAL vs joey_causal_ref), joey_parity2 (fixed, BL_OUT joey_out.jsonl).
- HF bf16 vs fp32 ref noise <=0.04 logits; HF bidir vs causal differ 0.15-1.7.
- probe kev08: AUROC conf 0.801 vs probe oof 0.742; confidence routing wins (logs/cascade/probe_kev08.out).
- decisions.rs unit test passes (cargo test -p kev --lib decisions).
- 16:04 PARITY PASS (fixed binary): joey ref25 100/100 max_dlogit 0.0016; elephant 16/16 max_dlogit 0.0018. per_decision_ms joey 169.5 elephant 593.0 (incl. CPU slot readout, single branch)
- 16:08 committed b800b1a (port), 0399a1f (decisions), 308e28e (probe)
- 16:10 queued tools/baselabs/energy.sh joey(100 items, reps 1/5) elephant(16, 2/10) x3 -> logs/baselabs/energy
- 16:14 queued tools/cascade/floor.sh (curl /v1/decisions, K8 vs K1 energy, idle srv vs none) -> logs/cascade/floor
- 16:25 probe kev4b: AUROC conf 0.817, probe dev 0.785, oof 0.739; kev4b alone 0.804 acc 18.9 J/dec beats always-large 0.782; best conf@10% 0.820 29.7 J/correct
- 16:30 causal control: 20/20 max_dlogit 0.0013 vs HF causal (engine exact; v1 error was entirely the attnfa redirect)
- 16:49 still queued on gpu_lock (other worker holds; flock not FIFO): joey_parity2, energy.sh x3, floor.sh, hidden kev9q8
- 23:26 rerun joey r1 r2, elephant r1 (gate timeouts), then probe kev9q8
- 23:30 floor: k8 0.66 J/dec vs k1 4.18 (6.3x); idle_none contaminated (15.2W > srv 8.65W) -> queued tools/cascade/idle.sh x3
- 23:35 RESULTS: joey-1 J/dec median 9.33 gross / 7.82 net (n=3, 154 ms/dec); elephant-1 42.26 / 36.20 (n=3, 672 ms; r1 outlier 61.1 at 102 W, likely contaminated). Parity in all energy runs 100/100 & 16/16.
  probe kev9q8: AUROC conf 0.810, probe dev 0.635, oof 0.758 -> confidence routing >= probe; Kev-9B alone (0.828) beats t2 (0.782).
  /v1/decisions curl OK (3 question types, empty questions -> 4xx error). k8 0.66 J/dec vs k1 4.18 (16 vs 115 ms/dec).
- Nov2 10:10Z: 27B rows+docs committed 195c2c8; idle r1-3 contaminated (moved to floor/contaminated_1030), requeued idle.sh (box load 25 from wai builds)
- Nov3 11:45Z: rivals chain3 done 07:54Z; analysis + b8 + ML.ENERGY rerun, rival tables in docs/decision_models.md. cbq (per-layer codebook) requant FAIL 09:04Z after layer 21 of 3-21 logged (act-err q3 2.45-2.74 vs cb 0.018-0.026); no exactness check ran, BEAM_CBLUT shader edits left uncommitted. Idle loop rerun started (1/6 clean so far).
