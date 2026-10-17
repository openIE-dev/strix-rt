//! Tiny spin-waiting fork-join pool for the latency-bound decoder (a few 10 µs jobs per TDT step;
//! spawning threads or condvar wake-ups per job would dominate).
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Mutex;

pub struct Pool<'a> { pub n: usize, epoch: AtomicU64, done: AtomicUsize, job: Mutex<Option<&'a (dyn Fn(usize) + Sync)>> }

impl<'a> Pool<'a> {
    /// Run `body` with a pool of `n` workers (worker 0 is the calling thread).
    pub fn scope<R>(n: usize, body: impl FnOnce(&Pool<'a>) -> R) -> R where 'a: 'a {
        let p = Pool { n, epoch: AtomicU64::new(0), done: AtomicUsize::new(0), job: Mutex::new(None) };
        std::thread::scope(|s| {
            for w in 1..n { let p = &p; s.spawn(move || {
                let mut seen = 0u64; let mut spins = 0u32;
                loop { let e = p.epoch.load(Acquire);
                    if e == u64::MAX { return }
                    if e != seen { seen = e; spins = 0; let j = p.job.lock().unwrap().unwrap(); j(w); p.done.fetch_add(1, Release); continue }
                    spins += 1; if spins < 1 << 16 { std::hint::spin_loop() } else { std::thread::yield_now() } }
            }); }
            let r = body(&p); p.epoch.store(u64::MAX, Release); r
        })
    }
    /// Run job(w) for every worker w in 0..n and wait for all of them.
    pub fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        // SAFETY: the reference only lives until every worker has finished this epoch (we wait below).
        let j: &'a (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(job) };
        *self.job.lock().unwrap() = Some(j); self.done.store(0, Release); self.epoch.fetch_add(1, AcqRel);
        job(0);
        while self.done.load(Acquire) < self.n - 1 { std::hint::spin_loop() }
    }
}

#[derive(Clone, Copy)] pub struct Out(pub *mut f32);
unsafe impl Send for Out {} unsafe impl Sync for Out {}

/// y[r] = b[r] + W[r].x (+ W2[r].x2) for rows r in this worker's slice. W row-major with `k` columns.
pub fn gemv_rows(w: usize, n: usize, rows: usize, k: usize, wm: &[f32], x: &[f32], w2: Option<(&[f32], &[f32])>, b: &[f32], y: Out) {
    let per = rows.div_ceil(n); let (r0, r1) = (w * per, ((w + 1) * per).min(rows));
    for r in r0..r1 { let mut s = b[r] + dot(&wm[r * k..(r + 1) * k], x);
        if let Some((w2, x2)) = w2 { s += dot(&w2[r * k..(r + 1) * k], x2); }
        unsafe { *y.0.add(r) = s } }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")] { if std::is_x86_feature_detected!("fma") { return unsafe { dot_fma(a, b) } } }
    let mut s = [0f32; 8]; for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) { for i in 0..8 { s[i] += ca[i] * cb[i]; } } s.iter().sum()
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len()) / 32 * 32; let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut s0, mut s1, mut s2, mut s3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0; while i < n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), s3); i += 32; }
    let s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut t = [0f32; 8]; _mm256_storeu_ps(t.as_mut_ptr(), s);
    let mut r: f32 = t.iter().sum(); for j in n..a.len().min(b.len()) { r += a[j] * b[j]; } r
}

pub fn env(n: &str, d: usize) -> usize { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
