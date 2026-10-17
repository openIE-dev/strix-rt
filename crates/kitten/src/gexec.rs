//! GPU execution of the decoder's heavy and elementwise ops on arena tensors (strided views read in place).
//! Returns Ok(None) for anything not handled, so the CPU interpreter takes it (after an automatic flush).
use crate::arena::{dispatch, touch};
use crate::dec::{bshape, dtype_of, rdt, DT, T, V};
use anyhow::Result;
const NONE: u32 = 0xffff_ffff;
thread_local! { pub static EPI: std::cell::Cell<Option<(u32, f32, (u32, isize, isize))>> = const { std::cell::Cell::new(None) }; }
/// linear with a fused epilogue: y = act((x W^T + b) / dv) + resid
pub fn linear_fused(a: &[V], epi: u32, dv: f32, resid: Option<&T>) -> Result<Option<V>> {
    let r = resid.map(|t| { touch(&t.s); let nd = t.nd(); (t.goff(), row_stride(t, nd - 1), t.st[nd - 1]) });
    if let Some((_, None, _)) = r { return Ok(None) }
    EPI.with(|e| e.set(Some((epi, dv, r.map_or((0, 0, 0), |(o, s, c)| (o, s.unwrap(), c))))));
    let out = exec("linear", a); EPI.with(|e| e.set(None)); out
}
thread_local! { static COOP: bool = std::env::var("DEC_COOP").map(|v| v != "0").unwrap_or(true); }
fn gk() -> usize { if COOP.with(|c| *c) { 4 } else { 1 } }

pub fn ew(op: u32, p0: f32, p1: f32, dst: &T, a: &T, b: Option<&T>) -> Result<()> {
    let nd = dst.nd(); let n = dst.numel(); if n == 0 { return Ok(()) }
    // collapse to <= 6 dims by dropping size-1 dims
    let ae = a.expand(&dst.sh); let be = b.map(|b| b.expand(&dst.sh));
    let dims: Vec<usize> = (0..nd).filter(|&i| dst.sh[i] != 1).collect();
    if dims.len() > 6 { anyhow::bail!("ew: >6 dims") }
    let mut sh = [1u32; 6]; let (mut ds, mut as_, mut bs) = ([0i32; 6], [0i32; 6], [0i32; 6]);
    for (j, &i) in dims.iter().enumerate() { sh[j] = dst.sh[i] as u32; ds[j] = dst.st[i] as i32; as_[j] = ae.st[i] as i32; if let Some(b) = &be { bs[j] = b.st[i] as i32 } }
    touch(&dst.s); touch(&a.s); if let Some(b) = b { touch(&b.s) }
    let mut p = vec![n as u32, dims.len().max(1) as u32, op, p0.to_bits(), p1.to_bits(), dst.goff(), ae.goff(), be.as_ref().map_or(0, |b| b.goff())];
    p.extend(sh); p.extend(ds.map(|x| x as u32)); p.extend(as_.map(|x| x as u32)); p.extend(bs.map(|x| x as u32));
    let g = n.div_ceil(256) as u32; let gx = g.min(65535);
    dispatch(0, &p, gx, g.div_ceil(gx), 1); Ok(())
}
fn contig(t: &T) -> Result<T> { if t.contig() { return Ok(t.clone()) } let o = T::empty(t.sh.clone(), t.dt); ew(0, 0.0, 0.0, &o, t, None)?; Ok(o) }
/// stride that walks all leading (row) dims as one, if they are mergeable
fn row_stride(t: &T, lead: usize) -> Option<isize> {
    let d: Vec<usize> = (0..lead).filter(|&i| t.sh[i] != 1).collect();
    if d.is_empty() { return Some(0) }
    for w in d.windows(2) { if t.st[w[0]] != t.st[w[1]] * t.sh[w[1]] as isize { return None } }
    Some(t.st[*d.last().unwrap()])
}
#[allow(clippy::too_many_arguments)]
fn gemm(m: usize, n: usize, k: usize, nb: usize, a: (u32, isize, isize, isize), b: (u32, isize, isize, isize), c: (u32, isize, isize, isize), bias: u32) {
    let p = [m as u32, n as u32, k as u32, 0, a.0, a.1 as u32, a.2 as u32, a.3 as u32, b.0, b.1 as u32, b.2 as u32, b.3 as u32, c.0, c.1 as u32, c.2 as u32, c.3 as u32, bias, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut p = p; let e = EPI.with(|e| e.take()); if let Some((epi, dv, r)) = e { p[25] = epi; p[26] = dv.to_bits(); p[27] = r.0; p[28] = r.1 as u32; p[29] = r.2 as u32; }
    let tiles = n.div_ceil(64) * m.div_ceil(64);
    if gk() == 4 && p[3] == 0 && nb == 1 && tiles < 160 && k >= 256 && SPLITK.with(|c| *c) {
        let want = 320usize.div_ceil(tiles).min(k / 128).max(2); let kc = k.div_ceil(want).div_ceil(32) * 32; let ns = k.div_ceil(kc);
        if ns > 1 { let tmp = T::empty(vec![ns * m * n], DT::F); touch(&tmp.s);
            let mut q = p.to_vec(); q.extend([ns as u32, kc as u32, tmp.goff(), B16.with(|b| b.take()).unwrap_or(0)]);
            dispatch(4, &q, n.div_ceil(64) as u32, m.div_ceil(64) as u32, ns as u32);
            q[31] = 0; dispatch(4, &q, n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1); return } }
    let mut q = p.to_vec(); q.extend([0, 0, 0, B16.with(|b| b.take()).unwrap_or(0)]);
    dispatch(gk(), &q, n.div_ceil(64) as u32, m.div_ceil(64) as u32, nb as u32);
}
thread_local! {
    static B16: std::cell::Cell<Option<u32>> = std::cell::Cell::new(None);
    static W16ON: bool = std::env::var("DEC_W16").map(|v| v != "0").unwrap_or(true);
    /// f32 weight store offset -> (packed f16 copy, aligned word offset), built once per weight
    static W16: std::cell::RefCell<std::collections::HashMap<u32, (T, u32, T)>> = Default::default();
}
fn w16(w: &T) -> u32 {
    W16.with(|c| { let mut c = c.borrow_mut(); let e = c.entry(w.goff()).or_insert_with(|| { let n = w.numel();
        let t = T::empty(vec![n / 2 + 4], DT::F); let wo = (t.goff() + 3) / 4 * 4; touch(&t.s); touch(&w.s);
        misc(9, n / 2, &[w.goff(), wo]); (t, wo, w.clone()) }); touch(&e.0.s); e.1 })
}
thread_local! { static SPLITK: bool = std::env::var("DEC_SPLITK").map(|v| v != "0").unwrap_or(false); }
fn misc(mode: u32, n: usize, u: &[u32]) { let mut p = vec![mode, n as u32]; p.extend(u); p.resize(24, 0); let g = n.div_ceil(256) as u32; let gx = g.min(65535); dispatch(6, &p, gx, g.div_ceil(gx), 1); }
fn cont_g(t: &T) -> Result<T> { let c = contig(t)?; touch(&c.s); Ok(c) }
pub fn exec(op: &str, a: &[V]) -> Result<Option<V>> {
    let t0 = || a[0].t();
    let un = |code: u32, p0: f32, p1: f32| -> Result<Option<V>> { let x = t0()?; let o = T::empty(x.sh.clone(), DT::F); ew(code, p0, p1, &o, &x, None)?; Ok(Some(V::T(o))) };
    let bin = |code: u32, p0: f32, dt: Option<DT>| -> Result<Option<V>> {
        let (x, y) = (t0()?, a[1].t()?); let dt = dt.unwrap_or_else(|| rdt(x.dt, a[1].sdt()));
        let o = T::empty(bshape(&x.sh, &y.sh), dt); ew(code, p0, 0.0, &o, &x, Some(&y))?; Ok(Some(V::T(o))) };
    // only take tensors that are big enough to matter, or already on the GPU path; tiny scalar math stays on CPU
    if let Some(V::T(_)) = a.first() { let t = a.iter().filter_map(|v| if let V::T(t) = v { Some(t) } else { None }).max_by_key(|t| t.numel()).unwrap(); if t.numel() < 64 && !matches!(op, "stft" | "istft" | "linear" | "matmul" | "_convolution" | "scaled_dot_product_attention" | "layer_norm" | "copy_" | "fill_" | "add_" | "mul_" | "cat") { return Ok(None) } } else if !matches!(op, "cat" | "zeros") { return Ok(None) }
    Ok(match op {
        "add" | "sub" | "rsub" => { let al = if a.len() > 2 { a[2].f()? as f32 } else { 1.0 }; bin(match op { "add" => 1, "sub" => 2, _ => 3 }, al, None)? }
        "add_" => { let (x, y) = (t0()?, a[1].t()?); let al = if a.len() > 2 { a[2].f()? as f32 } else { 1.0 }; ew(1, al, 0.0, &x, &x, Some(&y))?; Some(V::T(x)) }
        "mul_" => { let (x, y) = (t0()?, a[1].t()?); ew(4, 0.0, 0.0, &x, &x, Some(&y))?; Some(V::T(x)) }
        "copy_" | "fill_" => { let (x, y) = (t0()?, a[1].t()?); ew(0, 0.0, 0.0, &x, &y, None)?; Some(V::T(x)) }
        "mul" => bin(4, 0.0, None)?, "div" => bin(5, 0.0, Some(DT::F))?,
        "floor_divide" => bin(6, 0.0, None)?, "ge" => bin(7, 0.0, Some(DT::B))?, "gt" => bin(8, 0.0, Some(DT::B))?, "eq" => bin(9, 0.0, Some(DT::B))?,
        "masked_fill" => { let (x, mk) = (t0()?, a[1].t()?); let o = T::empty(bshape(&x.sh, &mk.sh), x.dt); ew(10, a[2].f()? as f32, 0.0, &o, &x, Some(&mk))?; Some(V::T(o)) }
        "pow" => un(11, a[1].f()? as f32, 0.0)?,
        "clamp" => un(12, if a[1].is_none() { f32::NEG_INFINITY } else { a[1].f()? as f32 }, if a[2].is_none() { f32::INFINITY } else { a[2].f()? as f32 })?,
        "clamp_min" => un(12, a[1].f()? as f32, f32::INFINITY)?,
        "abs" => un(13, 0.0, 0.0)?, "exp" => un(14, 0.0, 0.0)?, "sin" => un(15, 0.0, 0.0)?, "cos" => un(16, 0.0, 0.0)?, "tanh" => un(17, 0.0, 0.0)?,
        "silu" => un(18, 0.0, 0.0)?, "mish" => un(19, 0.0, 0.0)?,
        "gelu" => un(if matches!(a.get(1), Some(V::S(s)) if s == "tanh") { 21 } else { 20 }, 0.0, 0.0)?,
        "leaky_relu" => un(22, if a.len() > 1 { a[1].f()? as f32 } else { 0.01 }, 0.0)?,
        "elu" => un(23, a[1].f()? as f32, 0.0)?,
        "remainder" => un(24, a[1].f()? as f32, 0.0)?,
        "contiguous" => Some(V::T(contig(&t0()?)?)),
        "view" | "reshape" => { let x = t0()?; if x.contig() { return Ok(None) }
            { // split of the last dim of a row-strided view: stays a view
                let nd = x.nd(); let mut ns = a[1].il()?; let tot = x.numel() as i64; if let Some(p) = ns.iter().position(|&v| v == -1) { let r: i64 = ns.iter().filter(|&&v| v != -1).product(); ns[p] = tot / r.max(1) }
                if ns.len() == nd + 1 && x.st[nd - 1] == 1 && ns[..nd - 1].iter().zip(&x.sh[..nd - 1]).all(|(&a, &b)| a as usize == b) && (ns[nd - 1] * ns[nd]) as usize == x.sh[nd - 1] {
                    let mut st = x.st[..nd - 1].to_vec(); st.push(ns[nd] as isize); st.push(1); return Ok(Some(V::T(x.view(ns.iter().map(|&v| v as usize).collect(), st, x.off)))) } } Some(V::T(contig(&x)?.reshape(&a[1].il()?))) }
        "view_as" => { let x = t0()?; if x.contig() { return Ok(None) } let o = a[1].t()?; Some(V::T(contig(&x)?.reshape(&o.sh.iter().map(|&v| v as i64).collect::<Vec<_>>()))) }
        "zeros" => { let sh: Vec<usize> = a[0].il()?.iter().map(|&x| x as usize).collect(); let dt = if a[1].is_none() { DT::F } else { dtype_of(a[1].i()?) };
            let o = T::empty(sh, dt); ew(29, 0.0, 0.0, &o, &o, None)?; Some(V::T(o)) }
        "cat" => { let ts = a[0].tl()?; let nd = ts[0].nd() as i64; let d = { let d = a[1].i()?; if d < 0 { d + nd } else { d } } as usize;
            let mut sh = ts[0].sh.clone(); sh[d] = ts.iter().map(|t| t.sh[d]).sum();
            let o = T::empty(sh, ts.iter().fold(ts[0].dt, |p, t| rdt(p, t.dt))); let mut off = 0;
            for t in &ts { ew(0, 0.0, 0.0, &o.slice(d, off as i64, (off + t.sh[d]) as i64, 1), t, None)?; off += t.sh[d] } Some(V::T(o)) }
        "pad" if matches!(&a[2], V::S(s) if s == "reflect") => { let x = cont_g(&t0()?)?; let p = a[1].il()?; if p.len() != 2 { return Ok(None) }
            let n = *x.sh.last().unwrap(); let (l, r) = (p[0] as usize, p[1] as usize); let mut sh = x.sh.clone(); *sh.last_mut().unwrap() = n + l + r;
            let out = T::empty(sh, x.dt); touch(&out.s); misc(1, out.numel(), &[x.goff(), out.goff(), n as u32, l as u32, r as u32, 1, 0]); Some(V::T(out)) }
        "pad" => { let x = t0()?; let p = a[1].il()?; if p.len() != 2 || matches!(&a[2], V::S(s) if s != "constant") { return Ok(None) }
            let val = if a.len() > 3 && !a[3].is_none() { a[3].f()? as f32 } else { 0.0 }; let (l, r) = (p[0] as usize, p[1] as usize);
            let mut sh = x.sh.clone(); let ld = sh.len() - 1; sh[ld] += l + r; let o = T::empty(sh, x.dt); ew(29, val, 0.0, &o, &o, None)?;
            ew(0, 0.0, 0.0, &o.slice(ld, l as i64, (l + x.sh[ld]) as i64, 1), &x, None)?; Some(V::T(o)) }
        "to" => { let x = t0()?; let code = if a.len() == 8 { if a[1].is_none() { None } else { a[1].i().ok() } } else if matches!(a[1], V::S(_)) { a[2].i().ok() } else if a[1].is_none() { None } else { a[1].i().ok() };
            if std::env::var("TODBG").is_ok() { eprintln!("to {:?} {:?} code {:?} n{}", x.sh, x.dt as u8, code, a.len()) }
            match code.map(dtype_of) { Some(dt) if dt != x.dt => { let o = T::empty(x.sh.clone(), dt); ew(match dt { DT::B => 26, DT::I => 27, _ => 0 }, 0.0, 0.0, &o, &x, None)?; Some(V::T(o)) } _ => Some(V::T(x)) } }
        "bitwise_not" => { let x = t0()?; let o = T::empty(x.sh.clone(), DT::B); ew(25, 0.0, 0.0, &o, &x, None)?; Some(V::T(o)) }
        "linear" => {
            let (mut x, w) = (t0()?, a[1].t()?); let (n, k) = (w.sh[0], w.sh[1]); let nd = x.nd();
            let rs = match row_stride(&x, nd - 1) { Some(s) => s, None => { x = contig(&x)?; row_stride(&x, nd - 1).unwrap() } };
            let m = x.numel() / k; let mut sh = x.sh.clone(); sh[nd - 1] = n; let o = T::empty(sh, DT::F);
            let bias = if a.len() > 2 && !a[2].is_none() { let b = contig(&a[2].t()?)?; touch(&b.s); b.goff() } else { NONE };
            touch(&x.s); touch(&w.s); touch(&o.s);
            if gk() == 4 && W16ON.with(|c| *c) && k % 8 == 0 && w.contig() && w.nd() == 2 { let wo = w16(&w); B16.with(|b| b.set(Some(wo))); }
            gemm(m, n, k, 1, (x.goff(), rs, x.st[nd - 1], 0), (w.goff(), w.st[0], w.st[1], 0), (o.goff(), n as isize, 1, 0), bias); Some(V::T(o)) }
        "matmul" => {
            let (x, y) = (t0()?, a[1].t()?); if x.nd() < 2 || y.nd() < 2 { return Ok(None) }
            let (m, k, n) = (x.sh[x.nd() - 2], x.sh[x.nd() - 1], y.sh[y.nd() - 1]);
            let bsh = bshape(&x.sh[..x.nd() - 2], &y.sh[..y.nd() - 2]); let nb: usize = bsh.iter().product(); let nbd = bsh.len();
            let mut xs = bsh.clone(); xs.extend([m, k]); let mut ys = bsh.clone(); ys.extend([k, n]);
            let (mut xe, mut ye) = (x.expand(&xs), y.expand(&ys));
            let xb = match row_stride(&xe, nbd) { Some(s) => s, None => { xe = contig(&xe)?; row_stride(&xe, nbd).unwrap() } };
            let yb = match row_stride(&ye, nbd) { Some(s) => s, None => { ye = contig(&ye)?; row_stride(&ye, nbd).unwrap() } };
            let mut sh = bsh; sh.extend([m, n]); let o = T::empty(sh, DT::F); touch(&xe.s); touch(&ye.s); touch(&o.s);
            gemm(m, n, k, nb, (xe.goff(), xe.st[nbd], xe.st[nbd + 1], xb), (ye.goff(), ye.st[nbd + 1], ye.st[nbd], yb), (o.goff(), n as isize, 1, (m * n) as isize), NONE);
            Some(V::T(o)) }
        "_convolution" if a[6].b()? => {
            let (x, w) = (t0()?, cont_g(&a[1].t()?)?); let (s, p, g, op_) = (a[3].il()?[0] as usize, a[4].il()?[0] as usize, a[8].i()?, a[7].il()?[0] as usize);
            if g != 1 || a[5].il()?[0] != 1 { return Ok(None) }
            let (b, c, l) = (x.sh[0], x.sh[1], x.sh[2]); let (o, k) = (w.sh[1], w.sh[2]); let lo = (l - 1) * s + k - 2 * p + op_;
            let bias = if a[2].is_none() { NONE } else { cont_g(&a[2].t()?)?.goff() };
            let out = T::empty(vec![b, o, lo], DT::F); touch(&x.s); touch(&out.s);
            misc(0, b * o * lo, &[x.goff(), x.st[0] as u32, x.st[1] as u32, x.st[2] as u32, w.goff(), out.goff(), bias, c as u32, o as u32, l as u32, k as u32, s as u32, p as u32, lo as u32]);
            Some(V::T(out)) }
        "stft" => { let x = cont_g(&t0()?)?; let (nf, hop) = (a[1].i()? as usize, a[2].i()? as usize); if x.nd() != 2 || a[5].b().unwrap_or(false) || a[4].is_none() { return Ok(None) }
            let w = cont_g(&a[4].t()?)?; let (b, l) = (x.sh[0], x.sh[1]); let fr = 1 + (l - nf) / hop; let nb = nf / 2 + 1;
            let out = T::empty(vec![b, nb, fr, 2], DT::C); touch(&out.s);
            misc(2, b * nb * fr, &[x.goff(), out.goff(), w.goff(), l as u32, nf as u32, hop as u32, nb as u32, fr as u32]); Some(V::T(out)) }
        "istft" => { let x = cont_g(&t0()?)?; let (nf, hop) = (a[1].i()? as usize, a[2].i()? as usize); if x.nd() != 4 || a[4].is_none() { return Ok(None) }
            let w = cont_g(&a[4].t()?)?; let (b, nb, fr) = (x.sh[0], x.sh[1], x.sh[2]); let ol = hop * (fr - 1);
            let out = T::empty(vec![b, ol], DT::F); touch(&out.s);
            misc(3, b * ol, &[x.goff(), out.goff(), w.goff(), nf as u32, hop as u32, nb as u32, fr as u32, ol as u32]); Some(V::T(out)) }
        "_convolution" => {
            let (x, w) = (t0()?, contig(&a[1].t()?)?);
            let (s, p, d, g) = (a[3].il()?[0] as usize, a[4].il()?[0] as usize, a[5].il()?[0] as usize, a[8].i()? as usize);
            let (b, c, l) = (x.sh[0], x.sh[1], x.sh[2]); let (oc, cg, kw) = (w.sh[0], w.sh[1], w.sh[2]); let og = oc / g; let _ = c;
            let pf = PADF.with(|c| c.take()); let (p, lo) = match pf { Some((pl, pr)) if p == 0 => (pl, (l + pl + pr - d * (kw - 1) - 1) / s + 1), Some(_) => return Ok(None), None => (p, (l + 2 * p - d * (kw - 1) - 1) / s + 1) };
            let o = T::empty(vec![b, oc, lo], DT::F);
            let bias = if a[2].is_none() { NONE } else { let bb = contig(&a[2].t()?)?; touch(&bb.s); bb.goff() };
            touch(&x.s); touch(&w.s); touch(&o.s); let ep = EPI.with(|e| e.take()); if ep.is_some() && b != 1 { return Ok(None) }
            for bi in 0..b {
                let pp = [og as u32, lo as u32, (cg * kw) as u32, 1, w.goff(), (cg * kw) as u32, 1, (og * cg * kw) as u32,
                    (x.goff() as isize + bi as isize * x.st[0]) as u32, 0, 0, (cg as isize * x.st[1]) as u32,
                    o.goff() + (bi * oc * lo) as u32, lo as u32, 1, (og * lo) as u32, bias, og as u32, l as u32, kw as u32, s as u32, p as u32, d as u32, x.st[1] as u32, x.st[2] as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                let mut pp = pp; if let Some((epi, dv, r)) = ep { pp[25] = epi; pp[26] = dv.to_bits(); pp[27] = r.0; pp[28] = r.1 as u32; pp[29] = r.2 as u32; }
                dispatch(if std::env::var("DEC_COOP_CONV").map(|v| v != "0").unwrap_or(false) { gk() } else { 1 }, if std::env::var("DEC_COOP_CONV").map(|v| v != "0").unwrap_or(false) { &pp[..] } else { &pp[..30] }, lo.div_ceil(64) as u32, og.div_ceil(64) as u32, g as u32);
            }
            Some(V::T(o)) }
        "scaled_dot_product_attention" => {
            let (q, k, v) = (t0()?, a[1].t()?, a[2].t()?); if q.nd() != 4 || q.sh[3] != 64 || v.sh[3] != 64 || a[5].b()? { return Ok(None) }
            let (b, h, l, e) = (q.sh[0], q.sh[1], q.sh[2], q.sh[3]); let s = k.sh[2];
            let sc = if a.len() > 6 && !a[6].is_none() { a[6].f()? as f32 } else { 1.0 / (e as f32).sqrt() };
            let o = T::empty(vec![b, h, l, 64], DT::F);
            let (mo, mst, mt) = if a[3].is_none() { (0, [0isize; 4], 0u32) } else { let mk = a[3].t()?.expand(&[b, h, l, s]); touch(&mk.s); (mk.goff(), [mk.st[0], mk.st[1], mk.st[2], mk.st[3]], if mk.dt == DT::B { 2 } else { 1 }) };
            touch(&q.s); touch(&k.s); touch(&v.s); touch(&o.s);
            let st = |t: &T| [t.st[0] as u32, t.st[1] as u32, t.st[2] as u32, t.st[3] as u32];
            let mut p = vec![l as u32, s as u32, h as u32, sc.to_bits(), q.goff()]; p.extend(st(&q)); p.push(k.goff()); p.extend(st(&k)); p.push(v.goff()); p.extend(st(&v));
            p.extend([o.goff(), mo, mst[0] as u32, mst[1] as u32, mst[2] as u32, mst[3] as u32, mt]);
            if std::env::var("SDPA_DBG").is_ok() { eprintln!("SDPA q {:?} k {:?} v {:?} mt {mt} mst {mst:?}", q.st, k.st, v.st) }
            if std::env::var("SDPA_CM").map(|v| v != "0").unwrap_or(true) { p.extend([1, 1, 0, 0]); dispatch(5, &p, l.div_ceil(64) as u32, (b * h) as u32, 1); let mut o = o; o.st = vec![(l * h * 64) as isize, 64, (h * 64) as isize, 1]; return Ok(Some(V::T(o))) }
            let want: usize = std::env::var("SDPA_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
            let chunk = s.div_ceil(want).div_ceil(32) * 32; let ns = s.div_ceil(chunk);
            let tmp = T::empty(vec![b * h * l * ns * 66], DT::F); touch(&tmp.s);
            p.extend([ns as u32, chunk as u32, tmp.goff(), 0]);
            dispatch(2, &p, l.div_ceil(32) as u32, (b * h) as u32, ns as u32);
            if ns > 1 { let n = p.len(); p[n - 1] = 1; dispatch(2, &p, l.div_ceil(32) as u32, (b * h) as u32, 1); }
            Some(V::T(o)) }
        "embedding" => { let (w, idx) = (cont_g(&t0()?)?, cont_g(&a[1].t()?)?); if w.nd() != 2 { return Ok(None) } let d = w.sh[1];
            let mut sh = idx.sh.clone(); sh.push(d); let o = T::empty(sh, DT::F); touch(&o.s); misc(4, o.numel(), &[w.goff(), idx.goff(), d as u32, o.goff()]); Some(V::T(o)) }
        "upsample_nearest1d" => { let x = cont_g(&t0()?)?; if x.nd() != 3 || a[2].is_none() { return Ok(None) } let s = a[2].fl()?[0]; let l = x.sh[2]; let lo = (l as f64 * s).floor() as usize;
            let mut sh = x.sh.clone(); sh[2] = lo; let o = T::empty(sh, x.dt); touch(&o.s); misc(5, o.numel(), &[x.goff(), o.goff(), l as u32, lo as u32, ((1.0 / s) as f32).to_bits()]); Some(V::T(o)) }
        "sum" if a.len() < 2 || a[1].is_none() => { let x = cont_g(&t0()?)?; let o = T::empty(vec![], if x.dt == DT::F { DT::F } else { DT::I }); touch(&o.s); misc(6, 1, &[x.goff(), o.goff(), x.numel() as u32]); Some(V::T(o)) }
        "sum" => { let x = t0()?; let d = match a.get(1) { Some(V::L(_)) => a[1].il()?, _ => return Ok(None) }; if d.len() != 1 { return Ok(None) } let keep = a[2].b()?;
            let nd = x.nd(); let dd = if d[0] < 0 { (d[0] + nd as i64) as usize } else { d[0] as usize };
            let xt = cont_g(&x.transpose(dd, nd - 1))?; let n = x.sh[dd]; let mut sh = xt.sh.clone(); sh[nd - 1] = 1;
            let o = T::empty(sh, if x.dt == DT::F { DT::F } else { DT::I }); touch(&o.s); misc(6, o.numel(), &[xt.goff(), o.goff(), n as u32]); let r = o.transpose(dd, nd - 1); Some(V::T(if keep { r } else { r.squeeze(dd) })) }
        "cumsum" => { let x = cont_g(&t0()?)?; let nd = x.nd(); let d = a[1].i()?; if !(d == -1 || d as usize == nd - 1) { return Ok(None) }
            let n = x.sh[nd - 1]; let rows = x.numel() / n; let nc = n.div_ceil(256); let o = T::empty(x.sh.clone(), x.dt); let tmp = T::empty(vec![rows * nc], DT::F); touch(&o.s); touch(&tmp.s);
            let u = [x.goff(), o.goff(), n as u32, nc as u32, tmp.goff()]; misc(7, rows * nc, &u); misc(8, rows * nc, &u); Some(V::T(o)) }
        "repeat_interleave" => { let x = t0()?; if a.len() < 3 || a[2].is_none() { return Ok(None) } let r = a[1].i()? as usize; let d = a[2].i()?; if d != 0 { return Ok(None) }
            let mut sh = vec![x.sh[0], r]; sh.extend_from_slice(&x.sh[1..]); let mut st = vec![x.st[0], 0]; st.extend_from_slice(&x.st[1..]);
            let v = x.view(sh, st, x.off); let o = T::empty(v.sh.clone(), x.dt); ew(0, 0.0, 0.0, &o, &v, None)?;
            let mut osh: Vec<i64> = x.sh.iter().map(|&s| s as i64).collect(); osh[0] *= r as i64; Some(V::T(o.reshape(&osh))) }
        "complex" => { let (re, im) = (t0()?, a[1].t()?); if re.sh != im.sh { return Ok(None) } let mut sh = re.sh.clone(); sh.push(2); let o = T::empty(sh, DT::C); let n = o.nd();
            let half = |i: usize| { let mut v = o.select(n - 1, i as i64); v.dt = DT::F; v }; ew(0, 0.0, 0.0, &half(0), &re, None)?; ew(0, 0.0, 0.0, &half(1), &im, None)?; Some(V::T(o)) }
        "layer_norm" | "softmax" => {
            let mut x = t0()?; let nd = x.nd();
            if op == "softmax" { let d = a[1].i()?; if !(d == -1 || d as usize == nd - 1) { return Ok(None) } }
            if nd > 3 { x = contig(&x)?; }
            let dd = *x.sh.last().unwrap(); let rows = x.numel() / dd; let nd = x.nd();
            let (r1, sti, sto) = match nd { 1 => (1, 0, 0), 2 => (x.sh[0], x.st[0], 0), _ => (x.sh[nd - 2], x.st[nd - 2], x.st[nd - 3]) };
            let (w, bb, eps) = if op == "layer_norm" {
                let w = if a[2].is_none() { NONE } else { let t = contig(&a[2].t()?)?; touch(&t.s); t.goff() };
                let b = if a[3].is_none() { NONE } else { let t = contig(&a[3].t()?)?; touch(&t.s); t.goff() }; (w, b, a[4].f()? as f32) } else { (NONE, NONE, 0.0) };
            let o = T::empty(x.sh.clone(), DT::F); touch(&x.s); touch(&o.s);
            let p = [dd as u32, r1 as u32, (op == "softmax") as u32, eps.to_bits(), x.goff(), sto as u32, sti as u32, x.st[nd - 1] as u32, w, bb, o.goff(), 0, 0, 0];
            let gx = (rows as u32).min(65535); dispatch(3, &p, gx, (rows as u32).div_ceil(gx), 1); Some(V::T(o)) }
        _ => None,
    })
}

/// layer_norm over the last dim of a 3-d x, then transpose(1,2) -> mish -> * mask [-> + bias -> * mask]; returns [B, D, L] contiguous
pub fn layer_norm_mish(x: &T, w: &T, b: &T, eps: f32, mask: &T, bias2: Option<&T>) -> Result<T> {
    let (w, b, mk) = (cont_g(w)?, cont_g(b)?, cont_g(mask)?); let dd = x.sh[2]; let rows = x.sh[0] * x.sh[1];
    let o = T::empty(vec![x.sh[0], dd, x.sh[1]], DT::F); touch(&x.s); touch(&o.s);
    let b2 = match bias2 { Some(t) => { let t = cont_g(t)?; t.goff() } None => 0 };
    let p = [dd as u32, x.sh[1] as u32, 0, eps.to_bits(), x.goff(), x.st[0] as u32, x.st[1] as u32, x.st[2] as u32, w.goff(), b.goff(), o.goff(), 1 | if bias2.is_some() { 2 } else { 0 }, mk.goff(), b2];
    let gx = (rows as u32).min(65535); dispatch(3, &p, gx, (rows as u32).div_ceil(gx), 1); Ok(o)
}

/// fused snake activation x + c * sin(x * a)^2 with per-channel a, c of shape [1, C, 1]
pub fn snake(x: &T, a: &T, c: &T) -> Result<T> {
    let (x, a, c) = (cont_g(x)?, cont_g(a)?, cont_g(c)?); let o = T::empty(x.sh.clone(), DT::F); touch(&o.s);
    misc(11, x.numel(), &[x.goff(), a.goff(), c.goff(), o.goff(), x.sh[1] as u32, x.sh[2] as u32]); Ok(o)
}

/// conv1d with a fused residual add (out = conv(x) + r), batch 1
pub fn conv_resid(a: &[V], r: &T) -> Result<Option<V>> {
    touch(&r.s); let nd = r.nd(); let rs = match row_stride(r, nd - 1) { Some(s) => s, None => return Ok(None) };
    EPI.with(|e| e.set(Some((4, 1.0, (r.goff(), rs, r.st[nd - 1]))))); let out = exec("_convolution", a); EPI.with(|e| e.set(None)); out
}

thread_local! { static PADF: std::cell::Cell<Option<(usize, usize)>> = std::cell::Cell::new(None); }
/// conv1d on x with a zero pad (left, right) folded into the implicit-GEMM padding
pub fn conv_padded(a: &[V], pl: usize, pr: usize) -> Result<Option<V>> { PADF.with(|c| c.set(Some((pl, pr)))); let r = exec("_convolution", a); PADF.with(|c| c.set(None)); r }
