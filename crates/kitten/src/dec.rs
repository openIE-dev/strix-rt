//! CPU interpreter for the KittenTTS S3 decoder (flow-matching mel decoder + HiFT vocoder), executing the flat
//! TorchScript graph exported by ktexport.py (decoder.json + decoder.bin). Strided views with torch aliasing
//! semantics (slice/select/transpose + in-place copy_/fill_/add_/mul_), torch's CPU mt19937 RNG for randn/rand.
use crate::json::{self, J};
use anyhow::{anyhow, bail, Result};
use std::cell::RefCell;
use std::rc::Rc;
use crate::arena::{self, Store};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DT { F, I, B, C }
#[derive(Clone)]
pub struct T { pub s: Rc<Store>, pub sh: Vec<usize>, pub st: Vec<isize>, pub off: usize, pub dt: DT }
pub fn cstrides(sh: &[usize]) -> Vec<isize> { let mut st = vec![1isize; sh.len()]; for i in (0..sh.len().saturating_sub(1)).rev() { st[i] = st[i + 1] * sh[i + 1] as isize } st }
impl T {
    pub fn new(v: Vec<f32>, sh: Vec<usize>, dt: DT) -> T { debug_assert_eq!(v.len(), sh.iter().product::<usize>()); let t = T::empty(sh, dt); t.buf()[..v.len()].copy_from_slice(&v); t }
    pub fn empty(sh: Vec<usize>, dt: DT) -> T { let st = cstrides(&sh); T { s: Rc::new(arena::alloc(sh.iter().product())), sh, st, off: 0, dt } }
    pub fn buf(&self) -> &'static mut [f32] { arena::cpu_slice(&self.s) }
    pub fn goff(&self) -> u32 { (self.s.off + self.off) as u32 }
    pub fn scalar(x: f32, dt: DT) -> T { T::new(vec![x], vec![], dt) }
    pub fn numel(&self) -> usize { self.sh.iter().product() }
    pub fn nd(&self) -> usize { self.sh.len() }
    pub fn contig(&self) -> bool { let c = cstrides(&self.sh); self.sh.iter().zip(self.st.iter().zip(&c)).all(|(&n, (a, b))| n == 1 || a == b) }
    /// contiguous copy of the data (or a cheap clone of the slice when already contiguous)
    pub fn data(&self) -> Vec<f32> {
        let s = self.buf(); let n = self.numel();
        if self.contig() { return s[self.off..self.off + n].to_vec() }
        let mut out = Vec::with_capacity(n); strided_walk(&self.sh, &self.st, self.off, &mut |o| out.push(s[o])); out
    }
    pub fn item(&self) -> f32 { self.buf()[self.off] }
    pub fn cont(&self) -> T { if self.contig() && self.off == 0 { self.clone() } else { T::new(self.data(), self.sh.clone(), self.dt) } }
    pub fn with(&self, v: Vec<f32>) -> T { T::new(v, self.sh.clone(), self.dt) }
    pub fn view(&self, sh: Vec<usize>, st: Vec<isize>, off: usize) -> T { T { s: self.s.clone(), sh, st, off, dt: self.dt } }
    pub fn reshape(&self, shape: &[i64]) -> T {
        let n = self.numel(); let known: usize = shape.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
        let sh: Vec<usize> = shape.iter().map(|&d| if d < 0 { n / known.max(1) } else { d as usize }).collect();
        if self.contig() { let st = cstrides(&sh); self.view(sh, st, self.off) } else { T::new(self.data(), sh, self.dt) }
    }
    pub fn transpose(&self, a: usize, b: usize) -> T { let (mut sh, mut st) = (self.sh.clone(), self.st.clone()); sh.swap(a, b); st.swap(a, b); self.view(sh, st, self.off) }
    pub fn permute(&self, p: &[usize]) -> T { self.view(p.iter().map(|&i| self.sh[i]).collect(), p.iter().map(|&i| self.st[i]).collect(), self.off) }
    pub fn unsqueeze(&self, d: usize) -> T { let (mut sh, mut st) = (self.sh.clone(), self.st.clone()); let s = if d < st.len() { st[d] * sh[d] as isize } else { 1 }; sh.insert(d, 1); st.insert(d, s); self.view(sh, st, self.off) }
    pub fn squeeze(&self, d: usize) -> T { if self.sh.get(d) != Some(&1) { return self.clone() } let (mut sh, mut st) = (self.sh.clone(), self.st.clone()); sh.remove(d); st.remove(d); self.view(sh, st, self.off) }
    pub fn slice(&self, d: usize, s: i64, e: i64, step: i64) -> T {
        let n = self.sh[d] as i64; let fix = |x: i64| (if x < 0 { x + n } else { x }).clamp(0, n);
        let (s, e) = (fix(s), fix(e).max(fix(s)));
        let (mut sh, mut st) = (self.sh.clone(), self.st.clone()); sh[d] = ((e - s + step - 1) / step) as usize; st[d] *= step as isize;
        self.view(sh, st, (self.off as isize + s as isize * self.st[d]) as usize)
    }
    pub fn select(&self, d: usize, i: i64) -> T { let i = if i < 0 { i + self.sh[d] as i64 } else { i }; let t = self.slice(d, i, i + 1, 1); let (mut sh, mut st) = (t.sh.clone(), t.st.clone()); sh.remove(d); st.remove(d); t.view(sh, st, t.off) }
    pub fn expand(&self, sh: &[usize]) -> T {
        let k = sh.len() - self.nd(); let mut st = vec![0isize; sh.len()];
        for i in 0..self.nd() { st[k + i] = if self.sh[i] == 1 && sh[k + i] != 1 { 0 } else { self.st[i] } }
        self.view(sh.to_vec(), st, self.off)
    }
    /// in-place assignment of (broadcast) src into this view
    pub fn assign(&self, src: &T) {
        let src = src.expand(&self.sh); let v = src.data(); let s = self.buf(); let mut k = 0;
        strided_walk(&self.sh, &self.st, self.off, &mut |o| { s[o] = v[k]; k += 1 });
    }
}
fn strided_walk(sh: &[usize], st: &[isize], off: usize, f: &mut dyn FnMut(usize)) {
    let nd = sh.len(); if sh.iter().any(|&n| n == 0) { return }
    if nd == 0 { f(off); return }
    let mut idx = vec![0usize; nd]; let mut o = off as isize; let (ln, ls) = (sh[nd - 1], st[nd - 1]);
    loop {
        for j in 0..ln { f((o + j as isize * ls) as usize) }
        let mut d = nd - 1;
        loop { if d == 0 { return } d -= 1; idx[d] += 1; o += st[d]; if idx[d] < sh[d] { break } o -= st[d] * sh[d] as isize; idx[d] = 0; }
    }
}
pub fn bshape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len()); (0..n).map(|i| { let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 }; let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 }; if x == 1 { y } else { x } }).collect()
}
pub fn binop(a: &T, b: &T, dt: DT, f: impl Fn(f32, f32) -> f32 + Sync) -> T {
    let sh = bshape(&a.sh, &b.sh);
    let (mut x, y) = (a.expand(&sh).data(), b.expand(&sh).data());
    if x.len() < 1 << 16 { for (p, &q) in x.iter_mut().zip(&y) { *p = f(*p, q) } } else {
        let ch = x.len().div_ceil(nthreads()); let f = &f;
        std::thread::scope(|s| for (xc, yc) in x.chunks_mut(ch).zip(y.chunks(ch)) { s.spawn(move || for (p, &q) in xc.iter_mut().zip(yc) { *p = f(*p, q) }); }); }
    T::new(x, sh, dt)
}
pub fn unop(a: &T, f: impl Fn(f32) -> f32 + Sync) -> T { let mut v = a.data(); par_map(&mut v, &f); T::new(v, a.sh.clone(), DT::F) }
fn par_map(v: &mut [f32], f: &(impl Fn(f32) -> f32 + Sync)) {
    if v.len() < 1 << 16 { for x in v.iter_mut() { *x = f(*x) } return }
    let ch = v.len().div_ceil(nthreads()); std::thread::scope(|s| for c in v.chunks_mut(ch) { s.spawn(move || for x in c.iter_mut() { *x = f(*x) }); });
}
pub fn rdt(a: DT, b: DT) -> DT { if a == DT::F || b == DT::F { DT::F } else if a == DT::I || b == DT::I { DT::I } else { DT::B } }
pub fn nthreads() -> usize { static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *N.get_or_init(|| std::env::var("DEC_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8))) }

// ---------------------------------------------------------------- GEMM: C[M,N] = A[M,K] . B[N,K]^T (+bias[N])
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4(a: [&[f32]; 4], b: &[f32]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let k = b.len(); let mut acc = [_mm256_setzero_ps(); 4]; let mut i = 0;
    while i + 8 <= k { let bv = _mm256_loadu_ps(b.as_ptr().add(i)); for r in 0..4 { acc[r] = _mm256_fmadd_ps(_mm256_loadu_ps(a[r].as_ptr().add(i)), bv, acc[r]); } i += 8; }
    let mut out = [0f32; 4];
    for r in 0..4 { let mut t = [0f32; 8]; _mm256_storeu_ps(t.as_mut_ptr(), acc[r]); let mut s = ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7])); for j in i..k { s += a[r][j] * b[j] } out[r] = s; }
    out
}
pub fn gemm_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    let work = |rows: std::ops::Range<usize>, c: &mut [f32]| {
        let r0 = rows.start;
        for i in (rows.start..rows.end).step_by(4) {
            let ar: [&[f32]; 4] = std::array::from_fn(|r| { let ii = (i + r).min(rows.end - 1); &a[ii * k..ii * k + k] });
            for j in 0..n {
                let d = unsafe { dot4(ar, &b[j * k..j * k + k]) };
                for r in 0..4 { if i + r < rows.end { c[(i + r - r0) * n + j] = d[r] + bias.map_or(0.0, |bb| bb[j]) } }
            }
        }
    };
    let nt = nthreads().min((m * n * k / 200_000).max(1)).min(m.div_ceil(4));
    if nt <= 1 { work(0..m, &mut c); return c }
    let rows_per = m.div_ceil(nt).div_ceil(4) * 4;
    std::thread::scope(|s| for (t, cc) in c.chunks_mut(rows_per * n).enumerate() { let w = &work; s.spawn(move || w(t * rows_per..(t * rows_per + rows_per).min(m), cc)); });
    c
}
/// when M is small and N large (e.g. linear on a few rows), parallelise over N by swapping roles: C^T = B . A^T
fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, bias: Option<&[f32]>) -> Vec<f32> {
    if m >= 64 || n < 64 { return gemm_nt(a, b, m, n, k, bias) }
    let ct = gemm_nt(b, a, n, m, k, None); let mut c = vec![0f32; m * n];
    for i in 0..m { for j in 0..n { c[i * n + j] = ct[j * m + i] + bias.map_or(0.0, |bb| bb[j]) } } c
}

// ---------------------------------------------------------------- torch CPU RNG (mt19937, 24-bit float uniforms, normal_fill_16)
pub struct Mt { st: [u32; 624], i: usize }
impl Mt {
    pub fn new(seed: u32) -> Mt { let mut st = [0u32; 624]; st[0] = seed; for j in 1..624 { st[j] = 1812433253u32.wrapping_mul(st[j - 1] ^ (st[j - 1] >> 30)).wrapping_add(j as u32) } Mt { st, i: 624 } }
    pub fn next(&mut self) -> u32 {
        if self.i >= 624 { for k in 0..624 { let y = (self.st[k] & 0x8000_0000) | (self.st[(k + 1) % 624] & 0x7fff_ffff); let mut v = self.st[(k + 397) % 624] ^ (y >> 1); if y & 1 != 0 { v ^= 0x9908_b0df } self.st[k] = v } self.i = 0 }
        let mut y = self.st[self.i]; self.i += 1; y ^= y >> 11; y ^= (y << 7) & 0x9d2c_5680; y ^= (y << 15) & 0xefc6_0000; y ^ (y >> 18)
    }
    pub fn uniform(&mut self) -> f32 { ((self.next() & ((1 << 24) - 1)) as f32) * (1.0 / (1u32 << 24) as f32) }
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        assert!(n >= 16, "randn for <16 elements uses torch's scalar path (not implemented)");
        let mut d: Vec<f32> = (0..n).map(|_| self.uniform()).collect();
        let f16 = |d: &mut [f32]| for j in 0..8 { let u1 = 1.0 - d[j]; let u2 = d[j + 8]; let r = (-2.0 * u1.ln()).sqrt(); let th = 2.0 * std::f32::consts::PI * u2; d[j] = r * th.cos(); d[j + 8] = r * th.sin(); };
        let mut i = 0; while i + 16 <= n { f16(&mut d[i..i + 16]); i += 16 }
        if n % 16 != 0 { for j in n - 16..n { d[j] = self.uniform() } f16(&mut d[n - 16..]); }
        d
    }
}

// ---------------------------------------------------------------- values + graph
#[derive(Clone)]
pub enum V { T(T), I(i64), F(f64), B(bool), S(String), N, L(Vec<V>) }
impl V {
    pub fn t(&self) -> Result<T> { Ok(match self { V::T(t) => t.clone(), V::I(i) => T::scalar(*i as f32, DT::I), V::F(f) => T::scalar(*f as f32, DT::F), V::B(b) => T::scalar(*b as u8 as f32, DT::B), _ => bail!("expected tensor") }) }
    pub fn i(&self) -> Result<i64> { Ok(match self { V::I(i) => *i, V::F(f) => *f as i64, V::B(b) => *b as i64, V::T(t) => t.item() as i64, _ => bail!("expected int") }) }
    pub fn f(&self) -> Result<f64> { Ok(match self { V::I(i) => *i as f64, V::F(f) => *f, V::T(t) => t.item() as f64, _ => bail!("expected float") }) }
    pub fn b(&self) -> Result<bool> { Ok(match self { V::B(b) => *b, V::I(i) => *i != 0, _ => bail!("expected bool") }) }
    pub fn is_none(&self) -> bool { matches!(self, V::N) }
    pub fn il(&self) -> Result<Vec<i64>> { match self { V::L(v) => v.iter().map(|x| x.i()).collect(), V::I(i) => Ok(vec![*i]), _ => bail!("expected int list") } }
    pub fn fl(&self) -> Result<Vec<f64>> { match self { V::L(v) => v.iter().map(|x| x.f()).collect(), _ => bail!("expected float list") } }
    pub fn tl(&self) -> Result<Vec<T>> { match self { V::L(v) => v.iter().map(|x| x.t()).collect(), _ => bail!("expected tensor list") } }
    pub fn sdt(&self) -> DT { match self { V::I(_) => DT::I, V::B(_) => DT::B, V::T(t) => t.dt, _ => DT::F } }
}
struct Node { op: String, inp: Vec<usize>, out: Vec<usize>, c: Option<V> }
thread_local! { static FLUSHDBG: bool = std::env::var("DEC_FLUSHDBG").is_ok(); }
thread_local! { static CPUPROF: bool = std::env::var("DEC_CPUPROF").is_ok(); }
pub struct Decoder { is_const: Vec<bool>, nodes: Vec<Node>, n_values: usize, inputs: Vec<usize>, outputs: Vec<usize>, pub prof: RefCell<std::collections::HashMap<String, (f64, usize)>>, qkv: RefCell<std::collections::HashMap<usize, T>> }
/// output shape of linear(x, w): x.sh with last dim = w.sh[0]
fn a_shape(vals: &[V], inp: &[usize]) -> Vec<usize> { match (&vals[inp[0]], &vals[inp[1]]) { (V::T(x), V::T(w)) => { let mut s = x.sh.clone(); *s.last_mut().unwrap() = w.sh[0]; s } _ => vec![] } }
pub fn dtype_of(code: i64) -> DT { match code { 11 => DT::B, 0..=4 => DT::I, 9 | 10 => DT::C, _ => DT::F } }

impl Decoder {
    pub fn load(prefix: &str) -> Result<Decoder> {
        let j = json::parse(&std::fs::read_to_string(format!("{prefix}.json"))?)?;
        let bin = std::fs::read(format!("{prefix}.bin"))?;
        let fl = |off: usize, n: usize| -> Vec<f32> { bin[off * 4..(off + n) * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let mut nodes = vec![];
        for n in j.at("nodes")?.arr() {
            let op = n.at("op")?.str().to_string(); let out: Vec<usize> = n.at("out")?.i64s().iter().map(|&x| x as usize).collect();
            if op == "const" {
                let c = n.at("c")?; let (k, v) = c.obj().unwrap().iter().next().unwrap();
                let val = match k.as_str() {
                    "tensor" => { let a = v.arr(); let off = a[0].num() as usize; let sh: Vec<usize> = a[1].i64s().iter().map(|&x| x as usize).collect(); let n: usize = sh.iter().product();
                        let dt = match a[2].str() { "int64" | "int32" => DT::I, "bool" => DT::B, _ => DT::F }; V::T(T::new(fl(off, n), sh, dt)) }
                    "int" => V::I(v.num() as i64), "float" => V::F(v.num()), "bool" => V::B(matches!(v, J::B(true))), "str" => V::S(v.str().to_string()), "none" => V::N,
                    o => bail!("const kind {o}") };
                nodes.push(Node { op, inp: vec![], out, c: Some(val) });
            } else { nodes.push(Node { op, inp: n.at("in")?.i64s().iter().map(|&x| x as usize).collect(), out, c: None }); }
        }
        let nv = j.at("n_values")?.num() as usize; let mut is_const = vec![false; nv]; for n in &nodes { if n.c.is_some() { is_const[n.out[0]] = true } }
        Ok(Decoder { is_const, nodes, n_values: j.at("n_values")?.num() as usize, inputs: j.at("inputs")?.i64s().iter().map(|&x| x as usize).collect(), outputs: j.at("outputs")?.i64s().iter().map(|&x| x as usize).collect(), prof: Default::default(), qkv: Default::default() })
    }
    /// tokens: audio ids (+3 x 4299 padding already appended); returns 24 kHz audio
    pub fn run(&self, tokens: &[i64], prompt_token: &[i64], prompt_feat: &[f32], embedding: &[f32], seed: u32) -> Result<Vec<f32>> {
        let mut vals: Vec<V> = vec![V::N; self.n_values];
        let it = |v: &[i64]| V::T(T::new(v.iter().map(|&x| x as f32).collect(), vec![1, v.len()], DT::I));
        vals[self.inputs[1]] = it(tokens); vals[self.inputs[2]] = it(prompt_token);
        vals[self.inputs[3]] = V::T(T::new(prompt_feat.to_vec(), vec![1, prompt_feat.len() / 80, 80], DT::F));
        vals[self.inputs[4]] = V::T(T::new(embedding.to_vec(), vec![1, embedding.len()], DT::F));
        let mut rng = Mt::new(seed); let prof = std::env::var("DEC_PROF").is_ok();
        let mut last = vec![usize::MAX; self.n_values];
        for (ni, n) in self.nodes.iter().enumerate() { for &i in &n.inp { last[i] = ni } }
        for &o in &self.outputs { last[o] = usize::MAX }
        let gprof = std::env::var("DEC_GPROF").is_ok();
        let mut uses = vec![0u32; self.n_values]; for n in &self.nodes { for &i in &n.inp { uses[i] += 1 } }
        let fuse = arena::gpu().is_some() && std::env::var("DEC_FUSE").map(|v| v != "0").unwrap_or(true);
        let mut skip = vec![false; self.nodes.len()];
        let nxt = |i: usize| -> Option<usize> { (i + 1..self.nodes.len()).find(|&j| self.nodes[j].c.is_none() && !matches!(self.nodes[j].op.as_str(), "prim::ListConstruct" | "aten::size" | "aten::Int" | "prim::NumToTensor")) };
        for (ni, n) in self.nodes.iter().enumerate() {
            if let Some(c) = &n.c { vals[n.out[0]] = c.clone(); continue }
            if gprof { arena::LABEL.with(|l| { let mut l = l.borrow_mut(); l.clear(); l.push_str("F:"); l.push_str(n.op.strip_prefix("aten::").unwrap_or(&n.op)); if let Some(V::T(x)) = n.inp.first().map(|&i| &vals[i]) { l.push_str(&format!(" {:?}", x.sh)) } if let Some(V::T(w)) = n.inp.get(1).map(|&i| &vals[i]) { l.push_str(&format!("x{:?}", w.sh)) } }); }
            if skip[ni] { for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } } continue }
            // layer_norm -> transpose(1,2) -> mish -> mul(mask) [-> add_(bias) -> mul(mask)] in one rownorm dispatch
            if fuse && n.op == "aten::layer_norm" {
                let one = |v: usize| if uses[v] == 1 && last[v] != usize::MAX { Some(last[v]) } else { None };
                let mut done = false;
                if let (Some(k1), V::T(x)) = (one(n.out[0]), &vals[n.inp[0]]) { let c1 = &self.nodes[k1];
                    if x.nd() == 3 && x.sh[0] == 1 && c1.op == "aten::transpose" && matches!((&vals[c1.inp[1]], &vals[c1.inp[2]]), (V::I(1), V::I(2))) && !vals[n.inp[2]].is_none() && !vals[n.inp[3]].is_none() {
                    if let Some(k2) = one(c1.out[0]) { let c2 = &self.nodes[k2];
                    if c2.op == "aten::mish" { if let Some(k3) = one(c2.out[0]) { let c3 = &self.nodes[k3];
                    if c3.op == "aten::mul" && c3.inp[0] == c2.out[0] { if let V::T(mk) = &vals[c3.inp[1]] { if mk.sh == [1, 1, x.sh[1]] {
                        // optional bias + mask
                        let mut tail: Option<(usize, usize, T)> = None;
                        if let Some(k4) = one(c3.out[0]) { let c4 = &self.nodes[k4];
                            if c4.op == "aten::add_" && c4.inp[0] == c3.out[0] && matches!(vals.get(c4.inp[2]), Some(V::I(1))) { if let V::T(bt) = &vals[c4.inp[1]] { if bt.sh == [1, x.sh[2], 1] {
                                if let Some(k5) = one(c4.out[0]) { let c5 = &self.nodes[k5]; if c5.op == "aten::mul" && c5.inp[0] == c4.out[0] && c5.inp[1] == c3.inp[1] { tail = Some((k4, k5, bt.clone())) } } } } } }
                        let (w, b) = (vals[n.inp[2]].t()?, vals[n.inp[3]].t()?); let eps = vals[n.inp[4]].f()? as f32;
                        let o = crate::gexec::layer_norm_mish(x, &w, &b, eps, mk, tail.as_ref().map(|t| &t.2))?;
                        let mut ks = vec![k1, k2, k3]; if let Some((k4, k5, _)) = &tail { ks.push(*k4); ks.push(*k5); }
                        let outv = self.nodes[*ks.last().unwrap()].out[0];
                        if std::env::var("DEC_TRACE").is_ok() { eprintln!("FUSE ln-mish {ni} -> {ks:?}") }
                        vals[outv] = V::T(o); for &k in &ks { skip[k] = true }
                        for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } }
                        done = true;
                    } } } } } } } }
                if done { continue }
            }
            // snake: mul(x, a) -> sin -> pow(2) -> mul(c, .) -> add(x, ., 1)
            if fuse && n.op == "aten::mul" && uses[n.out[0]] == 1 {
                let one = |v: usize| if uses[v] == 1 && last[v] != usize::MAX { Some(last[v]) } else { None };
                let pc = |v: usize, xs: &T| matches!(&vals[v], V::T(t) if t.nd() == 3 && t.sh[0] == 1 && t.sh[2] == 1 && t.sh[1] == xs.sh[1]);
                if let V::T(x) = &vals[n.inp[0]] { if x.nd() == 3 && x.sh[0] == 1 && x.dt == DT::F && pc(n.inp[1], x) {
                    let chain = (|| { let k1 = one(n.out[0])?; let c1 = &self.nodes[k1]; if c1.op != "aten::sin" { return None }
                        let k2 = one(c1.out[0])?; let c2 = &self.nodes[k2]; if c2.op != "aten::pow" || !matches!(vals[c2.inp[1]], V::I(2)) { return None }
                        let k3 = one(c2.out[0])?; let c3 = &self.nodes[k3]; if c3.op != "aten::mul" || c3.inp[1] != c2.out[0] || !pc(c3.inp[0], x) { return None }
                        let k4 = one(c3.out[0])?; let c4 = &self.nodes[k4]; if c4.op != "aten::add" || c4.inp[0] != n.inp[0] || c4.inp[1] != c3.out[0] || !matches!(vals.get(c4.inp[2]), Some(V::I(1))) { return None }
                        Some((k1, k2, k3, k4)) })();
                    if let Some((k1, k2, k3, k4)) = chain {
                        let (a, c) = (vals[n.inp[1]].t()?, vals[self.nodes[k3].inp[0]].t()?);
                        let o = crate::gexec::snake(x, &a, &c)?; vals[self.nodes[k4].out[0]] = V::T(o);
                        for k in [k1, k2, k3, k4] { skip[k] = true }
                        for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } }
                        continue } } }
            }
            // constant-zero pad -> conv (pad 0): fold the pad into the conv
            if fuse && n.op == "aten::pad" && uses[n.out[0]] == 1 && last[n.out[0]] != usize::MAX {
                let k = last[n.out[0]]; let c = &self.nodes[k];
                let zero = n.inp.len() < 4 || vals[n.inp[3]].is_none() || matches!(vals[n.inp[3]], V::F(f) if f == 0.0) || matches!(vals[n.inp[3]], V::I(0));
                if c.op == "aten::_convolution" && c.inp[0] == n.out[0] && matches!(&vals[n.inp[2]], V::S(m) if m == "constant") && zero
                    && matches!(vals[c.inp[4]].il().ok().as_deref(), Some([0])) && matches!(vals[c.inp[3]].il().ok().as_deref(), Some([1])) && uses[c.out[0]] >= 1 {
                    if let (Ok(pp), V::T(x)) = (vals[n.inp[1]].il(), &vals[n.inp[0]]) { if pp.len() == 2 && pp[0] >= 0 && pp[1] >= 0 && x.nd() == 3 {
                        let mut a: Vec<V> = c.inp.iter().map(|&i| vals[i].clone()).collect(); a[0] = V::T(x.clone());
                        if let Some(r) = crate::gexec::conv_padded(&a, pp[0] as usize, pp[1] as usize)? {
                            vals[c.out[0]] = r; skip[k] = true;
                            for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } }
                            for &i in &c.inp { if last[i] == k && !self.is_const[i] { vals[i] = V::N } }
                            continue } } } } }
            // conv -> add(conv, r, 1) | add(r, conv, 1): residual fused into the conv epilogue
            if fuse && n.op == "aten::_convolution" && uses[n.out[0]] == 1 && last[n.out[0]] != usize::MAX {
                let k = last[n.out[0]]; let c = &self.nodes[k];
                if c.op == "aten::add" && c.inp.len() > 2 && matches!(vals.get(c.inp[2]), Some(V::I(1))) && (c.inp[0] == n.out[0]) != (c.inp[1] == n.out[0]) {
                    let ri = if c.inp[0] == n.out[0] { c.inp[1] } else { c.inp[0] };
                    if let (V::T(r), V::T(x), V::T(w)) = (&vals[ri], &vals[n.inp[0]], &vals[n.inp[1]]) {
                        if x.nd() == 3 && x.sh[0] == 1 && r.dt == DT::F && r.nd() == 3 && r.sh[0] == 1 && r.sh[1] == w.sh[0] {
                            let a: Vec<V> = n.inp.iter().map(|&i| vals[i].clone()).collect();
                            let r = r.clone();
                            if let Some(V::T(o)) = crate::gexec::conv_resid(&a, &r)? { if o.sh == r.sh {
                                vals[c.out[0]] = V::T(o); skip[k] = true;
                                for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } }
                                continue } else { bail!("conv_resid shape mismatch") } } } } } }
            // contiguous whose single consumer reads strided input and writes a fresh tensor: pass the view through
            if fuse && n.op == "aten::contiguous" && uses[n.out[0]] == 1 && last[n.out[0]] != usize::MAX {
                let c = &self.nodes[last[n.out[0]]];
                if matches!(c.op.as_str(), "aten::layer_norm" | "aten::linear" | "aten::mul" | "aten::add" | "aten::sub" | "aten::mish" | "aten::div") && c.inp[0] == n.out[0] {
                    vals[n.out[0]] = vals[n.inp[0]].clone(); for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } } continue } }
            // q/k/v: three bias-free linears on the same input -> one gemm with a cached concatenated weight; outputs are column views
            if fuse && n.op == "aten::linear" && ni + 2 < self.nodes.len() {
                let (n1, n2) = (&self.nodes[ni + 1], &self.nodes[ni + 2]);
                let nob = |m: &Node| m.inp.len() < 3 || vals[m.inp[2]].is_none();
                if n1.op == n.op && n2.op == n.op && n1.inp[0] == n.inp[0] && n2.inp[0] == n.inp[0] && nob(n) && nob(n1) && nob(n2) {
                    if let (V::T(w0), V::T(w1), V::T(w2)) = (&vals[n.inp[1]], &vals[n1.inp[1]], &vals[n2.inp[1]]) {
                        if w0.sh == w1.sh && w0.sh == w2.sh && w0.nd() == 2 {
                            let (on, k) = (w0.sh[0], w0.sh[1]);
                            let wm = self.qkv.borrow_mut().entry(ni).or_insert_with(|| { let m = T::empty(vec![3 * on, k], DT::F);
                                for (i, w) in [w0, w1, w2].iter().enumerate() { let _ = crate::gexec::ew(0, 0.0, 0.0, &m.slice(0, (i * on) as i64, ((i + 1) * on) as i64, 1), w, None); } m }).clone();
                            let a = vec![vals[n.inp[0]].clone(), V::T(wm), V::N];
                            if let Some(V::T(o)) = crate::gexec::exec("linear", &a)? {
                                let nd = o.nd(); let mut sh = o.sh.clone(); sh[nd - 1] = on;
                                for (i, m) in [n, n1, n2].iter().enumerate() { vals[m.out[0]] = V::T(o.view(sh.clone(), o.st.clone(), o.off + i * on)); }
                                skip[ni + 1] = true; skip[ni + 2] = true;
                                for &i in &n.inp { if last[i] <= ni + 2 && !self.is_const[i] { vals[i] = V::N } }
                                continue } } } } }
            if fuse && n.op == "aten::linear" {
                // linear -> [div by const scalar] -> [gelu | add residual], each intermediate used only by the next node
                let o0 = n.out[0]; let mut cur = o0; let mut epi = 0u32; let mut dv = 1f32; let mut resid: Option<T> = None; let mut chain = vec![]; let mut j = ni;
                let only = |v: usize, at: usize| last[v] == at;
                if let Some(k) = nxt(j) { let m = &self.nodes[k]; if m.op == "aten::div" && m.inp[0] == cur && only(cur, k) && self.is_const[m.inp[1]] {
                    if let Ok(d) = vals[m.inp[1]].f() { epi |= 1; dv = d as f32; cur = m.out[0]; chain.push(k); j = k; } } }
                if let Some(k) = nxt(j) { let m = &self.nodes[k];
                    if m.op == "aten::gelu" && m.inp[0] == cur && only(cur, k) && matches!(&vals[m.inp[1]], V::S(s) if s == "none") && epi == 0 { epi |= 2; cur = m.out[0]; chain.push(k); }
                    else if m.op == "aten::add" && m.inp[0] == cur && only(cur, k) && m.inp.len() > 2 && matches!(vals[m.inp[2]], V::I(1)) {
                        if let V::T(r) = &vals[m.inp[1]] { let osh = a_shape(&vals, &n.inp); if r.sh == osh && r.dt == DT::F { epi |= 4; resid = Some(r.clone()); cur = m.out[0]; chain.push(k); } } } }
                if !chain.is_empty() {
                    let a: Vec<V> = n.inp.iter().map(|&i| vals[i].clone()).collect();
                    if let Some(V::T(r)) = crate::gexec::linear_fused(&a, epi, dv, resid.as_ref())? {
                        vals[cur] = V::T(r); for &k in &chain { skip[k] = true }
                        for &i in &n.inp { if last[i] == ni && !self.is_const[i] { vals[i] = V::N } }
                        continue
                    }
                }
            }
            let a: Vec<V> = n.inp.iter().map(|&i| vals[i].clone()).collect();
            let t0 = std::time::Instant::now();
            if std::env::var("DEC_SHAPES").is_ok() && (n.op.contains("conv") || n.op.contains("scaled") || n.op.contains("linear") || n.op.contains("matmul")) {
                let sh: Vec<String> = a.iter().take(2).map(|v| if let V::T(t) = v { format!("{:?}", t.sh) } else { "-".into() }).collect(); eprintln!("SH {} {}", n.op, sh.join(" ")); }
            if std::env::var("DEC_TRACE").is_ok() { let sh: Vec<String> = a.iter().map(|v| match v { V::T(t) => format!("{:?}", t.sh), V::I(i) => format!("{i}"), V::F(f) => format!("{f}"), _ => "_".into() }).collect(); eprintln!("TR {ni} {} {:?} <- {:?} {}", n.op, n.out, n.inp, sh.join(" ")); }
            let r = self.exec(&n.op, &a, &mut rng).map_err(|e| anyhow!("node {ni} {}: {e}", n.op))?;
            if prof { arena::flush()?; let key = if n.op.contains("conv") || n.op.contains("linear") { if let (Some(V::T(x)), Some(V::T(w))) = (a.first(), a.get(1)) { format!("{} {:?}x{:?}", n.op, x.sh, w.sh) } else { n.op.clone() } } else { n.op.clone() }; let mut p = self.prof.borrow_mut(); let e = p.entry(key).or_default(); e.0 += t0.elapsed().as_secs_f64(); e.1 += 1; }
            if false { let mut p = self.prof.borrow_mut(); let e = p.entry(n.op.clone()).or_default(); e.0 += t0.elapsed().as_secs_f64(); e.1 += 1; }
            if n.out.len() != 1 { bail!("multi-output op {}", n.op) }
            vals[n.out[0]] = r;
            for &i in &n.inp { if last[i] == ni && !matches!(self.nodes.get(0), None) && !self.is_const[i] { vals[i] = V::N } }
            if let Ok(dump) = std::env::var("DEC_DUMP") { if dump.split(',').any(|d| d.parse() == Ok(n.out[0])) { if let V::T(t) = &vals[n.out[0]] { let d = t.data(); std::fs::write(format!("/tmp/dec_v{}.f32", n.out[0]), d.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?; } } }
        }
        let r = match &vals[self.outputs[0]] { V::T(t) => Ok(t.data()), _ => bail!("output not a tensor") }; drop(vals); arena::flush()?; r
    }
    fn exec(&self, op: &str, a: &[V], rng: &mut Mt) -> Result<V> {
        let op = op.strip_prefix("aten::").unwrap_or(op);
        if std::env::var("DEC_GPROF").is_ok() { arena::LABEL.with(|l| { let mut l = l.borrow_mut(); l.clear(); l.push_str(op); if matches!(op, "linear" | "_convolution") { if let (Some(V::T(x)), Some(V::T(w))) = (a.first(), a.get(1)) { l.push_str(&format!(" {:?}x{:?}", x.sh, w.sh)) } } else if std::env::var("DEC_GPROF").as_deref() == Ok("2") { if let Some(V::T(x)) = a.first() { l.push_str(&format!(" {:?}{:?}", x.sh, x.st)) } } }); }
        if arena::gpu().is_some() { if let Some(r) = crate::gexec::exec(op, a)? { return Ok(r) } }
        if CPUPROF.with(|c| *c) && a.iter().any(|v| matches!(v, V::T(t) if t.numel() >= 64)) {
            let t = std::time::Instant::now(); let r = self.exec_cpu(op, a, rng); let sh = a.iter().find_map(|v| if let V::T(t) = v { Some(t.sh.clone()) } else { None });
            let mut p = self.prof.borrow_mut(); let e = p.entry(format!("CPU {op} {:?}", sh.unwrap_or_default())).or_default(); e.0 += t.elapsed().as_secs_f64(); e.1 += 1; return r;
        }
        if FLUSHDBG.with(|c| *c) { let n0 = arena::gpu().map_or(0, |g| g.n_flush.get()); let r = self.exec_cpu(op, a, rng);
            if arena::gpu().map_or(0, |g| g.n_flush.get()) > n0 { let sh: Vec<String> = a.iter().map(|v| match v { V::T(t) => format!("T{:?}", t.sh), V::I(i) => format!("{i}"), _ => "_".into() }).collect(); eprintln!("FLUSH by {op} {}", sh.join(" ")); }
            return r }
        self.exec_cpu(op, a, rng)
    }
    fn exec_cpu(&self, op: &str, a: &[V], rng: &mut Mt) -> Result<V> {
        let t0 = || a[0].t();
        let dim = |t: &T, d: i64| -> usize { if d < 0 { (d + t.nd() as i64) as usize } else { d as usize } };
        let tv = |t: T| Ok(V::T(t));
        match op {
            "prim::ListConstruct" => Ok(V::L(a.to_vec())),
            "prim::NumToTensor" => Ok(V::T(a[0].t()?)),
            "Int" => Ok(V::I(a[0].i()?)),
            "ScalarImplicit" => { let t = t0()?; Ok(if t.dt == DT::F { V::F(t.item() as f64) } else { V::I(t.item() as i64) }) }
            "size" => { let t = t0()?; let d = dim(&t, a[1].i()?); Ok(V::I(t.sh[d] as i64)) }
            "contiguous" => tv(t0()?.cont()),
            "to" => {
                let t = t0()?;
                let code = match a.len() { 6 | 5 if matches!(a[1], V::S(_)) => a[2].i().ok(), _ => if matches!(a[1], V::N) { None } else { a[1].i().ok() } };
                // to.dtype_layout(self, dtype, layout, device, pin, non_blocking, copy, mf) has 8 args
                let code = if a.len() == 8 { if a[1].is_none() { None } else { a[1].i().ok() } } else { code };
                match code.map(dtype_of) { Some(dt) if dt != t.dt => { let v: Vec<f32> = t.data().iter().map(|&x| match dt { DT::B => (x != 0.0) as u8 as f32, DT::I => x.trunc(), _ => x }).collect(); tv(T::new(v, t.sh.clone(), dt)) } _ => tv(t) }
            }
            "transpose" => { let t = t0()?; let (x, y) = (dim(&t, a[1].i()?), dim(&t, a[2].i()?)); tv(t.transpose(x, y)) }
            "permute" => { let t = t0()?; let p: Vec<usize> = a[1].il()?.iter().map(|&d| dim(&t, d)).collect(); tv(t.permute(&p)) }
            "view" | "reshape" => tv(t0()?.reshape(&a[1].il()?)),
            "view_as" => { let o = a[1].t()?; tv(t0()?.reshape(&o.sh.iter().map(|&x| x as i64).collect::<Vec<_>>())) }
            "unsqueeze" => { let t = t0()?; let d = a[1].i()?; let d = if d < 0 { d + t.nd() as i64 + 1 } else { d }; tv(t.unsqueeze(d as usize)) }
            "squeeze" => { let t = t0()?; let d = dim(&t, a[1].i()?); tv(t.squeeze(d)) }
            "atleast_2d" => { let t = t0()?; tv(match t.nd() { 0 => t.unsqueeze(0).unsqueeze(0), 1 => t.unsqueeze(0), _ => t }) }
            "select" => { let t = t0()?; let d = dim(&t, a[1].i()?); tv(t.select(d, a[2].i()?)) }
            "slice" => { let t = t0()?; let d = dim(&t, a[1].i()?); let s = if a[2].is_none() { 0 } else { a[2].i()? }; let e = if a[3].is_none() { i64::MAX } else { a[3].i()? };
                tv(t.slice(d, s, e.min(t.sh[d] as i64 * 2 + 1), a[4].i()?)) }
            "expand" => { let t = t0()?; let s = a[1].il()?; let k = s.len() - t.nd(); let sh: Vec<usize> = s.iter().enumerate().map(|(i, &x)| if x < 0 { t.sh[i - k] } else { x as usize }).collect(); tv(t.expand(&sh)) }
            "expand_as" => { let o = a[1].t()?; tv(t0()?.expand(&o.sh)) }
            "view_as_real" => { let t = t0()?; let mut r = t.clone(); r.dt = DT::F; tv(r) }
            "complex" => { let (re, im) = (t0()?.data(), a[1].t()?.data()); let mut sh = a[0].t()?.sh.clone(); sh.push(2);
                tv(T::new(re.iter().zip(&im).flat_map(|(&x, &y)| [x, y]).collect(), sh, DT::C)) }
            "add" | "sub" | "add_" | "rsub" => {
                let (x, y) = (t0()?, a[1].t()?); let al = if a.len() > 2 { a[2].f()? as f32 } else { 1.0 };
                let dt = rdt(x.dt, a[1].sdt());
                let r = match op { "sub" => binop(&x, &y, dt, |p, q| p - al * q), "rsub" => binop(&x, &y, dt, |p, q| q - al * p), _ => binop(&x, &y, dt, |p, q| p + al * q) };
                if op == "add_" { x.assign(&r); tv(x) } else { tv(r) }
            }
            "mul" | "mul_" => { let (x, y) = (t0()?, a[1].t()?); let r = binop(&x, &y, rdt(x.dt, a[1].sdt()), |p, q| p * q); if op == "mul_" { x.assign(&r); tv(x) } else { tv(r) } }
            "div" => { let (x, y) = (t0()?, a[1].t()?); tv(binop(&x, &y, DT::F, |p, q| p / q)) }
            "floor_divide" => { let (x, y) = (t0()?, a[1].t()?); tv(binop(&x, &y, rdt(x.dt, y.dt), |p, q| (p / q).floor())) }
            "remainder" => { let x = t0()?; let m = a[1].f()? as f32; tv(x.with(x.data().iter().map(|&p| p - (p / m).floor() * m).collect())) }
            "pow" => { let x = t0()?; let e = a[1].f()? as f32; tv(unop(&x, |p| if e == 2.0 { p * p } else { p.powf(e) })) }
            "ge" => { let (x, y) = (t0()?, a[1].t()?); tv(binop(&x, &y, DT::B, |p, q| (p >= q) as u8 as f32)) }
            "gt" => { let (x, y) = (t0()?, a[1].t()?); tv(binop(&x, &y, DT::B, |p, q| (p > q) as u8 as f32)) }
            "eq" => { let (x, y) = (t0()?, a[1].t()?); tv(binop(&x, &y, DT::B, |p, q| (p == q) as u8 as f32)) }
            "bitwise_not" => { let x = t0()?; tv(T::new(x.data().iter().map(|&p| (p == 0.0) as u8 as f32).collect(), x.sh.clone(), DT::B)) }
            "masked_fill" => { let (x, m) = (t0()?, a[1].t()?); let v = a[2].f()? as f32; tv(binop(&x, &m, x.dt, move |p, q| if q != 0.0 { v } else { p })) }
            "clamp" | "clamp_min" => { let x = t0()?; let lo = if a[1].is_none() { f32::NEG_INFINITY } else { a[1].f()? as f32 }; let hi = if op == "clamp" && !a[2].is_none() { a[2].f()? as f32 } else { f32::INFINITY };
                tv(unop(&x, |p| p.max(lo).min(hi))) }
            "abs" => tv(unop(&t0()?, f32::abs)), "exp" => tv(unop(&t0()?, f32::exp)), "sin" => tv(unop(&t0()?, f32::sin)), "cos" => tv(unop(&t0()?, f32::cos)), "tanh" => tv(unop(&t0()?, f32::tanh)),
            "silu" => tv(unop(&t0()?, |x| x / (1.0 + (-x).exp()))),
            "mish" => tv(unop(&t0()?, |x| x * (if x > 20.0 { x } else { x.exp().ln_1p() }).tanh())),
            "gelu" => { let tanh = matches!(a.get(1), Some(V::S(s)) if s == "tanh"); tv(unop(&t0()?, move |x| if tanh { 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh()) } else { (0.5 * x as f64 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2))) as f32 })) }
            "leaky_relu" => { let s = if a.len() > 1 { a[1].f()? as f32 } else { 0.01 }; tv(unop(&t0()?, move |x| if x > 0.0 { x } else { x * s })) }
            "elu" => { let al = a[1].f()? as f32; tv(unop(&t0()?, move |x| if x > 0.0 { x } else { al * (x.exp() - 1.0) })) }
            "linalg_vector_norm" => { let x = t0()?; let d = a[2].il()?; if d.len() != 1 || dim(&x, d[0]) != x.nd() - 1 { bail!("norm dims") } let n = *x.sh.last().unwrap(); let v = x.data();
                let mut sh = x.sh.clone(); *sh.last_mut().unwrap() = 1; if !a[3].b()? { sh.pop(); } tv(T::new(v.chunks(n).map(|r| r.iter().map(|&p| p * p).sum::<f32>().sqrt()).collect(), sh, DT::F)) }
            "sum" => { let x = t0()?; let d = a[1].il()?; if d.len() != 1 { bail!("sum dims") } let dd = dim(&x, d[0]); let keep = a[2].b()?;
                let xt = x.transpose(dd, x.nd() - 1).cont(); let n = x.sh[dd]; let v = xt.data(); let mut sh = xt.sh.clone(); sh.pop();
                let r = T::new(v.chunks(n).map(|c| c.iter().sum()).collect(), sh, if x.dt == DT::F { DT::F } else { DT::I });
                let r = if keep { r.unsqueeze(r.nd()).transpose(dd, r.nd()) } else if dd == x.nd() - 1 { r } else { // move axis order back
                    let mut p: Vec<usize> = (0..r.nd()).collect(); let last = p.pop(); let _ = last; let mut q: Vec<usize> = (0..r.nd()).collect(); q.remove(r.nd() - 1); q.insert(dd, r.nd() - 1); let _ = p; r.permute(&q) }; tv(r) }
            "max" => { let x = t0()?; tv(T::scalar(x.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max), x.dt)) }
            "cumsum" => { let x = t0()?; let d = dim(&x, a[1].i()?); if d != x.nd() - 1 { bail!("cumsum dim") } let n = x.sh[d]; let mut v = x.data();
                for r in v.chunks_mut(n) { let mut s = 0f32; for p in r.iter_mut() { s += *p; *p = s } } tv(x.with(v)) }
            "softmax" => { let x = t0()?; let d = dim(&x, a[1].i()?); if d != x.nd() - 1 { bail!("softmax dim") } let n = x.sh[d]; let mut v = x.data(); for r in v.chunks_mut(n) { softmax(r) } tv(T::new(v, x.sh.clone(), DT::F)) }
            "arange" => { let e = a[0].f()?; let n = e.ceil().max(0.0) as usize; tv(T::new((0..n).map(|i| i as f32).collect(), vec![n], if a[0].sdt() == DT::F { DT::F } else { DT::I })) }
            "zeros" => { let sh: Vec<usize> = a[0].il()?.iter().map(|&x| x as usize).collect(); let dt = if a[1].is_none() { DT::F } else { dtype_of(a[1].i()?) }; tv(T::new(vec![0.0; sh.iter().product()], sh, dt)) }
            "fill_" => { let x = t0()?; x.assign(&a[1].t()?); tv(x) }
            "copy_" => { let x = t0()?; x.assign(&a[1].t()?); tv(x) }
            "randn" => { let sh: Vec<usize> = a[0].il()?.iter().map(|&x| x as usize).collect(); let n = sh.iter().product(); tv(T::new(rng.randn(n), sh, DT::F)) }
            "randn_like" => { let x = t0()?; tv(T::new(rng.randn(x.numel()), x.sh.clone(), DT::F)) }
            "rand" => { let sh: Vec<usize> = a[0].il()?.iter().map(|&x| x as usize).collect(); let n = sh.iter().product(); tv(T::new((0..n).map(|_| rng.uniform()).collect(), sh, DT::F)) }
            "cat" => { let ts = a[0].tl()?; let d = dim(&ts[0], a[1].i()?); let mut sh = ts[0].sh.clone(); sh[d] = ts.iter().map(|t| t.sh[d]).sum();
                let out = T::new(vec![0.0; sh.iter().product()], sh, ts.iter().fold(ts[0].dt, |p, t| rdt(p, t.dt))); let mut o = 0;
                for t in &ts { out.slice(d, o as i64, (o + t.sh[d]) as i64, 1).assign(t); o += t.sh[d] } tv(out) }
            "embedding" => { let (w, idx) = (t0()?.cont(), a[1].t()?); let d = w.sh[1]; let wd = w.data(); let ids = idx.data();
                let mut sh = idx.sh.clone(); sh.push(d); tv(T::new(ids.iter().flat_map(|&i| wd[i as usize * d..i as usize * d + d].iter().copied()).collect(), sh, DT::F)) }
            "repeat_interleave" => { let x = t0()?; let r = a[1].i()? as usize; let d = dim(&x, a[2].i()?); if d != 0 { bail!("repeat dim") }
                let parts: Vec<T> = (0..x.sh[0]).flat_map(|i| std::iter::repeat(x.slice(0, i as i64, i as i64 + 1, 1)).take(r)).collect();
                let mut sh = x.sh.clone(); sh[0] *= r; let out = T::new(vec![0.0; sh.iter().product()], sh, x.dt); for (i, p) in parts.iter().enumerate() { out.slice(0, i as i64, i as i64 + 1, 1).assign(p) } tv(out) }
            "upsample_nearest1d" => { let x = t0()?; let s = a[2].fl()?[0]; let l = x.sh[2]; let lo = (l as f64 * s).floor() as usize; let v = x.data();
                let mut sh = x.sh.clone(); sh[2] = lo; let sc = (1.0 / s) as f32;
                tv(T::new(v.chunks(l).flat_map(|r| (0..lo).map(move |i| r[((i as f32 * sc).floor() as usize).min(l - 1)])).collect(), sh, x.dt)) }
            "layer_norm" => { let x = t0()?; let d = a[1].il()?; if d.len() != 1 { bail!("ln shape") } let n = d[0] as usize; let eps = a[4].f()? as f32;
                let w = if a[2].is_none() { None } else { Some(a[2].t()?.data()) }; let b = if a[3].is_none() { None } else { Some(a[3].t()?.data()) };
                let mut v = x.data(); for r in v.chunks_mut(n) { let m = r.iter().sum::<f32>() / n as f32; let var = r.iter().map(|&p| (p - m) * (p - m)).sum::<f32>() / n as f32; let is = 1.0 / (var + eps).sqrt();
                    for (i, p) in r.iter_mut().enumerate() { *p = (*p - m) * is * w.as_ref().map_or(1.0, |w| w[i]) + b.as_ref().map_or(0.0, |b| b[i]) } }
                tv(T::new(v, x.sh.clone(), DT::F)) }
            "linear" => { let (x, w) = (t0()?, a[1].t()?); let (n, k) = (w.sh[0], w.sh[1]); let m = x.numel() / k; let b = if a.len() > 2 && !a[2].is_none() { Some(a[2].t()?.data()) } else { None };
                let wd = w.cont(); let ws = wd.buf(); let y = gemm(&x.data(), &ws[wd.off..wd.off + n * k], m, n, k, b.as_deref());
                let mut sh = x.sh.clone(); *sh.last_mut().unwrap() = n; tv(T::new(y, sh, DT::F)) }
            "matmul" => { let (x, y) = (t0()?, a[1].t()?); if x.nd() < 2 || y.nd() < 2 { bail!("matmul 1d") }
                let (m, k, n) = (x.sh[x.nd() - 2], x.sh[x.nd() - 1], y.sh[y.nd() - 1]);
                let bsh = bshape(&x.sh[..x.nd() - 2], &y.sh[..y.nd() - 2]); let nb: usize = bsh.iter().product();
                let mut xs = bsh.clone(); xs.extend([m, k]); let mut ys = bsh.clone(); ys.extend([k, n]);
                let xd = x.expand(&xs).data(); let yt = y.expand(&ys); let yd = yt.transpose(yt.nd() - 2, yt.nd() - 1).data();
                let mut out = Vec::with_capacity(nb * m * n); for bi in 0..nb { out.extend(gemm(&xd[bi * m * k..][..m * k], &yd[bi * n * k..][..n * k], m, n, k, None)) }
                let mut sh = bsh; sh.extend([m, n]); tv(T::new(out, sh, DT::F)) }
            "scaled_dot_product_attention" => { let (q, k, v) = (t0()?, a[1].t()?, a[2].t()?); let mask = if a[3].is_none() { None } else { Some(a[3].t()?) };
                if a[5].b()? { bail!("causal sdpa") } let e = *q.sh.last().unwrap(); let sc = if a.len() > 6 && !a[6].is_none() { a[6].f()? as f32 } else { 1.0 / (e as f32).sqrt() };
                tv(sdpa(&q, &k, &v, mask.as_ref(), sc)) }
            "pad" => { let x = t0()?; let p = a[1].il()?; let refl = matches!(&a[2], V::S(s) if s == "reflect"); let val = if a.len() > 3 && !a[3].is_none() { a[3].f()? as f32 } else { 0.0 };
                if p.len() != 2 { bail!("pad len") } let (l, r) = (p[0] as usize, p[1] as usize); let n = *x.sh.last().unwrap(); let v = x.data(); let no = n + l + r;
                let mut sh = x.sh.clone(); *sh.last_mut().unwrap() = no;
                tv(T::new(v.chunks(n).flat_map(|row| (0..no).map(move |i| { let j = i as i64 - l as i64; if j >= 0 && (j as usize) < n { row[j as usize] } else if refl { let jj = if j < 0 { -j } else { 2 * (n as i64 - 1) - j }; row[jj as usize] } else { val } })).collect(), sh, x.dt)) }
            "_convolution" => { let (x, w) = (t0()?, a[1].t()?); let b = if a[2].is_none() { None } else { Some(a[2].t()?.data()) };
                let (s, p, d, tr, g) = (a[3].il()?[0] as usize, a[4].il()?[0] as usize, a[5].il()?[0] as usize, a[6].b()?, a[8].i()? as usize);
                tv(if tr { conv_t(&x, &w, b.as_deref(), s, p, g, a[7].il()?[0] as usize)? } else { conv1d(&x, &w, b.as_deref(), s, p, d, g)? }) }
            "stft" => { let x = t0()?; let (nf, hop) = (a[1].i()? as usize, a[2].i()? as usize); let w = a[4].t()?.data(); if a[5].b().unwrap_or(false) { bail!("normalized stft") }
                tv(stft(&x, nf, hop, &w)) }
            "istft" => { let x = t0()?; let (nf, hop) = (a[1].i()? as usize, a[2].i()? as usize); let w = a[4].t()?.data(); tv(istft(&x, nf, hop, &w)) }
            o => bail!("unsupported op {o}"),
        }
    }
}
extern "C" { #[link_name = "erf"] fn c_erf(x: f64) -> f64; }
fn erf(x: f64) -> f64 { unsafe { c_erf(x) } }
fn softmax(r: &mut [f32]) { let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max); if m == f32::NEG_INFINITY { return } let mut s = 0.0; for p in r.iter_mut() { *p = (*p - m).exp(); s += *p } for p in r.iter_mut() { *p /= s } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sdpa_rows(q: &[f32], k: &[f32], v: &[f32], e: usize, s: usize, ev: usize, sc: f32, mrow: &dyn Fn(usize, usize) -> f32, mbool: bool, rows: std::ops::Range<usize>, out: &mut [f32]) {
    let mut p = vec![0f32; s];
    for (oi, i) in rows.enumerate() {
        let qi = &q[i * e..i * e + e];
        for j in 0..s { let kj = &k[j * e..j * e + e]; let mut d = 0f32; for t in 0..e { d += qi[t] * kj[t] } let mut x = d * sc; let mv = mrow(i, j); if mbool { if mv == 0.0 { x = f32::NEG_INFINITY } } else { x += mv } p[j] = x; }
        softmax(&mut p);
        let o = &mut out[oi * ev..oi * ev + ev]; o.fill(0.0);
        for j in 0..s { let pj = p[j]; if pj == 0.0 { continue } let vj = &v[j * ev..j * ev + ev]; for t in 0..ev { o[t] += pj * vj[t] } }
    }
}
fn sdpa(q: &T, k: &T, v: &T, mask: Option<&T>, sc: f32) -> T {
    // q [B,H,L,E] k [B,H,S,E] v [B,H,S,Ev]; mask broadcast to [B,H,L,S] (bool: true = keep, float: additive), read through strides
    let (b, h, l, e) = (q.sh[0], q.sh[1], q.sh[2], q.sh[3]); let s = k.sh[2]; let ev = v.sh[3];
    let (qd, kd, vd) = (q.data(), k.data(), v.data());
    let mv = mask.map(|m| m.expand(&[b, h, l, s]));
    let mdata = mv.as_ref().map(|m| m.buf().to_vec()); let (mst, moff, mbool) = mv.as_ref().map_or((vec![0; 4], 0, false), |m| (m.st.clone(), m.off, m.dt == DT::B));
    let mut out = vec![0f32; b * h * l * ev];
    let nt = nthreads(); let rc = (b * h * l).div_ceil(nt * 4).max(1);
    std::thread::scope(|scope| for (ci, chunk) in out.chunks_mut(rc * ev).enumerate() {
        let (qd, kd, vd, mdata, mst) = (&qd, &kd, &vd, &mdata, &mst);
        scope.spawn(move || { let r0 = ci * rc; let r1 = (r0 + rc).min(b * h * l); let mut done = 0;
            let mut r = r0; while r < r1 { let bh = r / l; let i0 = r % l; let i1 = (i0 + (r1 - r)).min(l);
                let (bi, hi) = (bh / h, bh % h);
                let mf = |i: usize, j: usize| -> f32 { match mdata { Some(md) => md[(moff as isize + bi as isize * mst[0] + hi as isize * mst[1] + i as isize * mst[2] + j as isize * mst[3]) as usize], None => 0.0 } };
                unsafe { sdpa_rows(&qd[bh * l * e..][..l * e], &kd[bh * s * e..][..s * e], &vd[bh * s * ev..][..s * ev], e, s, ev, sc, &mf, mbool, i0..i1, &mut chunk[done * ev..(done + i1 - i0) * ev]) }
                done += i1 - i0; r += i1 - i0; } });
    });
    T::new(out, vec![b, h, l, ev], DT::F)
}
fn conv1d(x: &T, w: &T, bias: Option<&[f32]>, s: usize, p: usize, d: usize, g: usize) -> Result<T> {
    let (b, c, l) = (x.sh[0], x.sh[1], x.sh[2]); let (o, cg, k) = (w.sh[0], w.sh[1], w.sh[2]); if cg * g != c { bail!("conv groups") }
    let lo = (l + 2 * p - d * (k - 1) - 1) / s + 1; let xd = x.data(); let wd = w.data(); let og = o / g;
    let mut out = vec![0f32; b * o * lo];
    for bi in 0..b { for gi in 0..g {
        // cols[lo, cg*k]
        let mut cols = vec![0f32; lo * cg * k];
        for t in 0..lo { for ci in 0..cg { for kk in 0..k { let pos = (t * s + kk * d) as i64 - p as i64; if pos >= 0 && (pos as usize) < l { cols[t * cg * k + ci * k + kk] = xd[bi * c * l + (gi * cg + ci) * l + pos as usize] } } } }
        let y = gemm(&wd[gi * og * cg * k..][..og * cg * k], &cols, og, lo, cg * k, None);
        for oi in 0..og { for t in 0..lo { out[bi * o * lo + (gi * og + oi) * lo + t] = y[oi * lo + t] + bias.map_or(0.0, |bb| bb[gi * og + oi]) } }
    } }
    Ok(T::new(out, vec![b, o, lo], DT::F))
}
fn conv_t(x: &T, w: &T, bias: Option<&[f32]>, s: usize, p: usize, g: usize, op: usize) -> Result<T> {
    if g != 1 { bail!("grouped conv_transpose") }
    let (b, c, l) = (x.sh[0], x.sh[1], x.sh[2]); let (o, k) = (w.sh[1], w.sh[2]); let lo = (l - 1) * s + k - 2 * p + op;
    // Y[o*k, l] = W^T[o*k, c] . X[c, l]  -> gemm_nt(A = W^T [o*k, c], B = X^T [l, c])
    let wd = w.data(); let mut wt = vec![0f32; o * k * c]; for ci in 0..c { for oi in 0..o { for kk in 0..k { wt[(oi * k + kk) * c + ci] = wd[(ci * o + oi) * k + kk] } } }
    let xd = x.data(); let mut out = vec![0f32; b * o * lo];
    for bi in 0..b {
        let mut xt = vec![0f32; l * c]; for ci in 0..c { for t in 0..l { xt[t * c + ci] = xd[bi * c * l + ci * l + t] } }
        let y = gemm(&wt, &xt, o * k, l, c, None);
        for oi in 0..o { for kk in 0..k { for t in 0..l { let pos = (t * s + kk) as i64 - p as i64; if pos >= 0 && (pos as usize) < lo { out[bi * o * lo + oi * lo + pos as usize] += y[(oi * k + kk) * l + t] } } } }
        if let Some(bb) = bias { for oi in 0..o { for t in 0..lo { out[bi * o * lo + oi * lo + t] += bb[oi] } } }
    }
    Ok(T::new(out, vec![b, o, lo], DT::F))
}
fn twiddle(nf: usize) -> (Vec<f64>, Vec<f64>) { ((0..nf).map(|i| (2.0 * std::f64::consts::PI * i as f64 / nf as f64).cos()).collect(), (0..nf).map(|i| (2.0 * std::f64::consts::PI * i as f64 / nf as f64).sin()).collect()) }
fn stft(x: &T, nf: usize, hop: usize, w: &[f32]) -> T {
    let (b, l) = (x.sh[0], x.sh[1]); let fr = 1 + (l - nf) / hop; let nb = nf / 2 + 1; let xd = x.data(); let (ct, st) = twiddle(nf);
    let mut out = vec![0f32; b * nb * fr * 2]; let mut fbuf = vec![0f64; nf];
    for bi in 0..b { for t in 0..fr { for n in 0..nf { fbuf[n] = (xd[bi * l + t * hop + n] * w[n]) as f64 }
        for f in 0..nb { let (mut re, mut im) = (0f64, 0f64); for n in 0..nf { let k = (f * n) % nf; re += fbuf[n] * ct[k]; im -= fbuf[n] * st[k]; }
            let o = ((bi * nb + f) * fr + t) * 2; out[o] = re as f32; out[o + 1] = im as f32; } } }
    T::new(out, vec![b, nb, fr, 2], DT::C)
}
fn istft(x: &T, nf: usize, hop: usize, w: &[f32]) -> T {
    // x complex [B, nb, frames, 2]; center=True, onesided
    let (b, nb, fr) = (x.sh[0], x.sh[1], x.sh[2]); let xd = x.data(); let ol = nf + hop * (fr - 1); let (ct, st) = twiddle(nf);
    let mut out = vec![0f32; b * (ol - nf)];
    for bi in 0..b {
        let mut y = vec![0f64; ol]; let mut env = vec![0f64; ol];
        for t in 0..fr { for n in 0..nf { let mut s = 0f64;
            for f in 0..nb { let o = ((bi * nb + f) * fr + t) * 2; let (re, im) = (xd[o] as f64, xd[o + 1] as f64); let k = (f * n) % nf;
                let edge = f == 0 || (nf % 2 == 0 && f == nf / 2); s += if edge { re * ct[k] } else { 2.0 * (re * ct[k] - im * st[k]) }; }
            y[t * hop + n] += s / nf as f64 * w[n] as f64; env[t * hop + n] += (w[n] * w[n]) as f64; } }
        for i in 0..ol - nf { let j = i + nf / 2; out[bi * (ol - nf) + i] = (y[j] / if env[j] > 1e-11 { env[j] } else { 1.0 }) as f32 }
    }
    T::new(out, vec![b, ol - nf], DT::F)
}
