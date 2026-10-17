//! CPU reference validation:
//!  1) Prism PQ2_0 / PTQ1_0 and Kitten TQ2_1 rows vs the forks' own ggml to_float (bit-exact expected)
//!  2) PQ2_0 vs PTQ1_0 codes+scales of the same Bonsai tensor (same checkpoint -> identical weights)
//!  3) Kitten flat 5-trit safetensors vs Kitten TQ2_1 GGUF (codes identical, scales bf16 vs f16)
//!  4) parakeet thrush rows: every byte < 243 and per-module zero fraction == ternary.json
//!  5) FWHT butterfly vs explicit normalized Sylvester matrix
use anyhow::Result;
use ternary::{dequant_row, fwht_ref, gg::Gg, st::St, thrush_row, thrush_row_flat, Fmt};
fn home() -> String { std::env::var("HOME").unwrap() }
fn fork_rows(tool: &str, path: &str, name: &str, r0: usize, nr: usize) -> Result<Vec<f32>> {
    let out = format!("/tmp/tern_fork_{}.f32", std::process::id());
    let o = std::process::Command::new(tool).args([path, name, &r0.to_string(), &nr.to_string(), &out]).output()?;
    anyhow::ensure!(o.status.success(), "fork tool failed: {}", String::from_utf8_lossy(&o.stderr));
    let b = std::fs::read(&out)?; Ok(b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}
fn mine(g: &Gg, name: &str, r0: usize, nr: usize) -> Vec<f32> {
    let t = g.tensor(name).unwrap(); let f = Fmt::from_ggml(t.ty).unwrap(); let k = t.dims[0] as usize; let rb = f.row_bytes(k);
    let d = g.data(t); let mut y = vec![0f32; k * nr];
    for r in 0..nr { dequant_row(f, &d[(r0 + r) * rb..(r0 + r + 1) * rb], &mut y[r * k..(r + 1) * k]); } y
}
fn main() -> Result<()> {
    let h = home();
    let pq2 = format!("{h}/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf"); let ptq1 = format!("{h}/models/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf");
    let kit = format!("{h}/models/kitten2/model-tq2_1.gguf");
    let mut all_ok = true;
    let cases: Vec<(String, String, Vec<&str>)> = vec![
        ("/tmp/fork_dequant_prism-llama".into(), pq2.clone(), vec!["blk.0.attn_qkv.weight", "blk.0.ffn_down.weight", "blk.3.attn_q.weight", "blk.63.ffn_up.weight", "blk.62.ssm_out.weight", "output.weight", "token_embd.weight"]),
        ("/tmp/fork_dequant_prism-llama".into(), ptq1.clone(), vec!["blk.0.attn_qkv.weight", "blk.0.ffn_down.weight", "blk.3.attn_q.weight", "blk.63.ffn_up.weight", "blk.62.ssm_out.weight", "output.weight", "token_embd.weight"]),
        ("/tmp/fork_dequant_kitten-cpp".into(), kit.clone(), vec!["blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.attn_k.weight", "blk.27.attn_output.weight"]),
    ];
    println!("## 1) CPU reference vs fork ggml to_float");
    for (tool, path, names) in &cases {
        let g = Gg::open(path)?;
        for n in names { let t = g.tensor(n).unwrap(); let rows = t.dims[1] as usize;
            let mut maxd = 0f32; let mut cnt = 0usize; let mut nz = 0usize;
            for r0 in [0, rows / 3, rows / 2, rows - 8] { let a = fork_rows(tool, path, n, r0, 8)?; let b = mine(&g, n, r0, 8);
                for (x, y) in a.iter().zip(&b) { maxd = maxd.max((x - y).abs()); if x.to_bits() != y.to_bits() { cnt += 1; } if *y != 0.0 { nz += 1; } } }
            let ok = cnt == 0; all_ok &= ok;
            println!("{} {n} [{}x{}] ty={} : 32 rows, bit-mismatches={cnt} max|d|={maxd:e} nonzero={nz} {}", path.rsplit('/').next().unwrap(), t.dims[1], t.dims[0], t.ty, if ok { "OK" } else { "FAIL" });
        }
    }
    println!("## 2) PQ2_0 vs PTQ1_0 (same checkpoint)");
    { let a = Gg::open(&pq2)?; let b = Gg::open(&ptq1)?;
      for n in ["blk.0.attn_qkv.weight", "blk.40.ffn_gate.weight", "output.weight"] { let rows = a.tensor(n).unwrap().dims[1] as usize; let mut diff = 0usize; let mut tot = 0usize;
        for r0 in [0, rows / 2, rows - 4] { let x = mine(&a, n, r0, 4); let y = mine(&b, n, r0, 4); tot += x.len(); diff += x.iter().zip(&y).filter(|(p, q)| p.to_bits() != q.to_bits()).count(); }
        println!("{n}: {diff}/{tot} values differ {}", if diff == 0 { "OK (identical)" } else { "DIFF" }); all_ok &= diff == 0; } }
    println!("## 3) Kitten flat 5-trit safetensors vs TQ2_1 GGUF");
    { let s = St::open(&format!("{h}/models/kitten2/model-ternary.safetensors"))?; let g = Gg::open(&kit)?;
      for (sn, gn) in [("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"), ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"), ("model.layers.13.self_attn.k_proj.weight", "blk.13.attn_k.weight")] {
        let codes = s.data(s.get(&format!("{sn}::codes")).unwrap()); let sct = s.get(&format!("{sn}::scale")).unwrap(); let sc = s.f32s(sct);
        let (rows, groups) = (sct.shape[0], sct.shape[1]); let k = groups * 128;
        let gt = g.tensor(gn).unwrap(); assert_eq!((gt.dims[0] as usize, gt.dims[1] as usize), (k, rows));
        let mut sign_diff = 0usize; let mut maxrel = 0f32; let mut tot = 0usize;
        for r in [0, 1, rows / 2, rows - 1] { let mut y = vec![0f32; k]; thrush_row_flat(codes, r, k, &sc[r * groups..(r + 1) * groups], &mut y); let z = mine(&g, gn, r, 1);
            for (a, b) in y.iter().zip(&z) { tot += 1; if a.signum() != b.signum() || (*a == 0.0) != (*b == 0.0) { sign_diff += 1; } if *b != 0.0 { maxrel = maxrel.max(((a - b) / b).abs()); } } }
        println!("{sn} [{rows}x{k}] vs {gn}: code mismatches {sign_diff}/{tot}, max rel scale diff {maxrel:.3e} (bf16 vs f16 scales) {}", if sign_diff == 0 { "OK" } else { "FAIL" }); all_ok &= sign_diff == 0; } }
    println!("## 4) parakeet-redux thrush-ternary-v2");
    { let s = St::open(&format!("{h}/models/parakeet-redux/model.safetensors"))?; let tj = std::fs::read_to_string(format!("{h}/models/parakeet-redux/ternary.json"))?;
      let mut nmod = 0; let mut worst = 0f64; let mut bad_bytes = 0usize; let mut bytes = 0usize;
      for chunk in tj.split("\"name\": \"").skip(1) { let name = chunk.split('"').next().unwrap();
        let zf: f64 = chunk.split("\"zero_fraction\": ").nth(1).unwrap().split(|c: char| c == '\n' || c == '}' || c == ',').next().unwrap().trim().parse()?;
        let (Some(q), Some(sc)) = (s.get(&format!("{name}.qweight")), s.get(&format!("{name}.scales"))) else { println!("missing {name}"); continue };
        let (n, rb) = (q.shape[0], q.shape[1]); let k = sc.shape[1] * 128; let k = if rb * 5 >= k { k } else { rb * 5 };
        let d = s.data(q); let scs = s.f32s(sc); let g = sc.shape[1]; let mut zeros = 0usize; let mut y = vec![0f32; k];
        for r in 0..n { thrush_row(&d[r * rb..(r + 1) * rb], &scs[r * g..(r + 1) * g], &mut y); zeros += y.iter().filter(|v| **v == 0.0).count(); }
        bad_bytes += d.iter().filter(|b| **b >= 243).count(); bytes += d.len();
        let z = zeros as f64 / (n * k) as f64; worst = worst.max((z - zf).abs()); nmod += 1; }
      println!("{nmod} modules: bytes>=243: {bad_bytes}/{bytes}; max |zero_fraction - ternary.json| = {worst:.2e} {}", if bad_bytes == 0 && worst < 1e-5 { "OK" } else { "CHECK" }); }
    println!("## 5) FWHT butterfly vs explicit matrix (n=1024)");
    { let n = 1024; let mut s = 1u64; let x: Vec<f32> = (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 40) as f32 / 16777216.0) - 0.5 }).collect();
      let a = fwht_ref(&x, None, n, None); let mut m = 0f32;
      for r in 0..n { let v: f64 = (0..n).map(|c| if ((r & c) as u32).count_ones() & 1 == 1 { -(x[c] as f64) } else { x[c] as f64 }).sum::<f64>() / (n as f64).sqrt(); m = m.max((v as f32 - a[r]).abs()); }
      println!("max abs diff {m:e} {}", if m < 1e-5 { "OK" } else { "FAIL" }); }
    println!("ALL {}", if all_ok { "OK" } else { "FAILURES" });
    Ok(())
}
