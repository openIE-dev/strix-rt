//! Port of kitten-tts-2-cpp text.cpp: normalize() wrapper rules, split() into <= 380-char chunks, join() of chunk waves.
//! Each regex of the C++ file is hand-coded as a matcher (leftmost, greedy, non-overlapping like std::regex_replace).
//! The FST backend (kitten_text_processing: numbers, dates, units) is NOT ported: normalize() behaves like the fork with the
//! backend unavailable (its rules still run; spoken-form expansion is skipped).
type C = Vec<char>;
fn ws(c: char) -> bool { c.is_whitespace() }
fn s2c(s: &str) -> C { s.chars().collect() }
fn c2s(c: &[char]) -> String { c.iter().collect() }
/// replace every leftmost match; `f(s, i)` returns (match length, replacement) for a match starting at i
fn rep(s: &[char], f: impl Fn(&[char], usize) -> Option<(usize, String)>) -> C {
    let mut out = vec![]; let mut i = 0;
    while i < s.len() { match f(s, i) { Some((n, r)) if n > 0 => { out.extend(r.chars()); i += n } _ => { out.push(s[i]); i += 1 } } }
    out
}
fn trim(s: &[char]) -> C { let a = s.iter().position(|&c| !ws(c)).unwrap_or(s.len()); let b = s.iter().rposition(|&c| !ws(c)).map_or(a, |p| p + 1); s[a..b.max(a)].to_vec() }
fn run(s: &[char], i: usize, p: impl Fn(char) -> bool) -> usize { let mut j = i; while j < s.len() && p(s[j]) { j += 1 } j }
const TAGS1: [&str; 10] = ["mundane", "nervous", "tender", "angry", "excited", "stern", "sad", "contemplative", "surprised", "joyful"];
const TAGS2: [&str; 10] = ["pause", "sigh", "gasp", "laugh", "giggle", "sob", "scoff", "growl", "um", "gulp"];
fn m_tag(s: &[char], i: usize) -> Option<usize> {
    let (open, close, set) = match s[i] { '[' => ('[', ']', &TAGS1), '<' => ('<', '>', &TAGS2), _ => return None }; let _ = open;
    let j = s[i + 1..].iter().position(|&c| c == close)? + i + 1; let w: String = c2s(&s[i + 1..j]).to_lowercase();
    set.contains(&w.as_str()).then_some(j + 1 - i)
}
fn m_emph(s: &[char], i: usize) -> Option<usize> {
    if s.len() < i + 7 || s[i..i + 3] != ['(', '(', '('] { return None }
    let j = run(s, i + 3, |c| c != '(' && c != ')' && c != '\n'); let n = j - i - 3;
    (n >= 1 && n <= 80 && s.len() >= j + 3 && s[j..j + 3] == [')', ')', ')']).then_some(j + 3 - i)
}
pub fn has_expression(t: &str) -> bool { let s = s2c(t); (0..s.len()).any(|i| m_tag(&s, i).is_some() || m_emph(&s, i).is_some()) }
fn dashes(s: &[char]) -> C {
    let s = rep(s, |s, i| { let j = run(s, i, ws); let k = if j < s.len() && (s[j] == '\u{2014}' || s[j] == '\u{2013}') { j + 1 } else if j + 1 < s.len() && s[j] == '-' && s[j + 1] == '-' { j + 2 } else { return None }; Some((run(s, k, ws) - i, ", ".into())) });
    let s = rep(&s, |s, i| { if s[i] != ',' { return None } let j = run(s, i + 1, ws); let k = run(s, j, |c| c == ','); (k > j).then(|| (k - i, ", ".into())) });
    let s = rep(&s, |s, i| { if !ws(s[i]) { return None } let j = run(s, i, ws); (j < s.len() && ",.!?;:".contains(s[j])).then(|| (j + 1 - i, s[j].to_string())) });
    let s = rep(&s, |s, i| { if s[i] != ',' { return None } let j = run(s, i + 1, ws); (j < s.len() && ".!?".contains(s[j])).then(|| (j + 1 - i, s[j].to_string())) });
    let s = rep(&s, |s, i| { let j = run(s, i, ws); (j - i >= 2).then(|| (j - i, " ".into())) });
    let mut s = s; let j = run(&s, 0, ws); if j < s.len() && s[j] == ',' { let k = run(&s, j + 1, ws); s.drain(..k); }
    trim(&s)
}
fn terminate(s: &[char]) -> C {
    let mut s = rep(s, |s, i| { if !ws(s[i]) { return None } let j = run(s, i, ws); (j < s.len() && "?!".contains(s[j])).then(|| (j + 1 - i, s[j].to_string())) });
    if let Some(p) = s.iter().position(|c| c.is_alphabetic()) { let u: C = s[p].to_uppercase().collect(); if u.len() == 1 { s[p] = u[0] } }
    let s = trim(&s);
    if s.is_empty() || ".!?\"\u{2019}\u{201d}')]".contains(*s.last().unwrap()) { return s }
    let mut e = s.len(); let t = s.iter().rposition(|&c| !ws(c)).map_or(0, |p| p + 1);
    let mut k = t; while k > 0 && "-,;:\u{2013}\u{2014}".contains(s[k - 1]) { k -= 1 } if k < t { e = k }
    let mut s = trim(&s[..e]); if s.is_empty() { return s } s.push('.'); s
}
pub fn normalize(input: &str) -> String {
    let mut saved: Vec<String> = vec![];
    let mut s = s2c(input);
    for pass in 0..2 { let mut out = vec![]; let mut i = 0;
        while i < s.len() { let m = if pass == 0 { m_tag(&s, i) } else { m_emph(&s, i) };
            match m { Some(n) => { saved.push(c2s(&s[i..i + n])); out.extend(format!("\u{e000}{}\u{e001}", saved.len() - 1).chars()); i += n } None => { out.push(s[i]); i += 1 } } }
        s = out; }
    s = rep(&s, |s, i| { if !ws(s[i]) { return None } let j = run(s, i, ws); s[i..j].contains(&'\n').then(|| (j - i, " ".into())) });
    s = rep(&s, |s, i| { (i + 2 < s.len() && s[i].is_ascii_lowercase() && ".!?;:".contains(s[i + 1]) && s[i + 2].is_ascii_uppercase()).then(|| (2, format!("{}{} ", s[i], s[i + 1]))) });
    s = rep(&s, |s, i| {
        if s[i] == '\u{2026}' { return Some((1, ", ".into())) }
        if s[i] != '.' { return None }
        let (mut j, mut g) = (i, 0); loop { if j < s.len() && s[j] == '.' { let k = run(s, j + 1, ws); if k > j + 1 { g += 1; j = k; continue } } break }
        // backtrack to the largest group count >= 2 that is followed by '.'
        if g >= 2 { let mut ends = vec![]; let mut k = i; for _ in 0..g { k = run(s, k + 1, ws); ends.push(k) } for gi in (2..=g).rev() { let e = ends[gi - 1]; if e < s.len() && s[e] == '.' { return Some((e + 1 - i, ", ".into())) } } }
        (i + 2 < s.len() && s[i + 1] == '.' && s[i + 2] == '.').then(|| (3, ", ".into())) });
    let mut m = vec![]; for &c in &s { match crate::math_table::math(c) { Some(r) => m.extend(r.chars()), None => m.push(c) } } s = m;
    s = rep(&s, |s, i| { let j = run(s, i, |c| c == ' ' || c == '\t'); (j - i >= 2).then(|| (j - i, " ".into())) });
    let word = |c: char| c.is_alphanumeric() || c == '_';
    s = rep(&s, |s, i| { if !(s[i] == 'v' || s[i] == 'V') || (i > 0 && word(s[i - 1])) { return None }
        let mut j = run(s, i + 1, |c| c.is_ascii_digit()); if j == i + 1 { return None } let mut groups = 0;
        while j + 1 < s.len() && s[j] == '.' && s[j + 1].is_ascii_digit() { j = run(s, j + 1, |c| c.is_ascii_digit()); groups += 1 }
        (groups > 0).then(|| (j - i, format!("{} {}", s[i], c2s(&s[i + 1..j]).replace('.', " point ")))) });
    s = rep(&s, |s, i| { if !s[i].is_ascii_digit() { return None } let j = run(s, i + 1, ws); (j < s.len() && "?!".contains(s[j]) && (j + 1 == s.len() || ws(s[j + 1]))).then(|| (j + 1 - i, format!("{} {}", s[i], s[j]))) });
    s = rep(&s, |s, i| { let (close, bad, max): (char, &[char], usize) = match s[i] { '[' => (']', &['[', ']', '\n'], 400), '<' => ('>', &['<', '>', '\n'], 60), '(' => (')', &['(', ')', '\n'], 400), _ => return None };
        let j = run(s, i + 1, |c| !bad.contains(&c)); let n = j - i - 1; (n >= 1 && n <= max && j < s.len() && s[j] == close).then(|| (j + 1 - i, format!(", {},", c2s(&trim(&s[i + 1..j]))))) });
    loop { let n = rep(&s, |s, i| { if s[i] != ',' { return None } let j = run(s, i + 1, ws); (j < s.len() && ",.!?;:".contains(s[j])).then(|| (j + 1 - i, s[j].to_string())) }); if n == s { break } s = n; }
    s = rep(&s, |s, i| { let j = run(s, i, |c| "\u{2022}\u{2023}\u{2043}\u{25e6}\u{25b8}\u{21b3}\u{2192}\u{2190}\u{2191}\u{2193}".contains(c)); (j > i).then(|| (j - i, " ".into())) });
    s = dashes(&s);
    if !trim(&s).is_empty() { s = terminate(&s); }
    let mut r = c2s(&s); for (i, v) in saved.iter().enumerate() { r = r.replace(&format!("\u{e000}{i}\u{e001}"), v); }
    r
}
pub fn split(text: &str, max_chars: usize, min_chars: usize) -> Vec<String> {
    let s = trim(&s2c(text)); if s.is_empty() { return vec![] }
    if max_chars == 0 || s.len() <= max_chars { return vec![c2s(&s)] }
    let mut sentences: Vec<C> = vec![]; let mut pos = 0; let mut i = 0;
    while i < s.len() { if ".!?".contains(s[i]) { let j = run(&s, i + 1, |c| "\"\u{2019}\u{201d}')]".contains(c)); let k = run(&s, j, ws);
            if k > j { sentences.push(trim(&s[pos..i + 1])); pos = k; i = k; continue } } i += 1; }
    if pos < s.len() { sentences.push(trim(&s[pos..])); }
    let mut parts: Vec<C> = vec![]; let mut cur: C = vec![];
    for mut sen in sentences {
        while sen.len() > max_chars {
            let head = &sen[..max_chars]; let mut best = 0;
            for sep in [", ", "; ", " \u{2014} "] { let p = s2c(sep); if let Some(f) = (0..=head.len().saturating_sub(p.len())).rev().find(|&k| head[k..k + p.len()] == p[..]) { best = best.max(f) } }
            let cut = if best > max_chars / 3 { best + 1 } else { max_chars };
            if !cur.is_empty() { parts.push(std::mem::take(&mut cur)); }
            parts.push(trim(&sen[..cut])); sen = trim(&sen[cut..]);
        }
        if cur.is_empty() { cur = sen } else if cur.len() + 1 + sen.len() <= max_chars { cur.push(' '); cur.extend(sen) } else { parts.push(std::mem::replace(&mut cur, sen)) }
    }
    if !cur.is_empty() { parts.push(cur) }
    let lim = (max_chars as f64 * 1.4) as usize; let mut changed = true;
    while changed && parts.len() > 1 && min_chars > 0 { changed = false;
        for i in 0..parts.len() { if parts[i].len() >= min_chars { continue }
            let prev = i > 0 && parts[i - 1].len() + 1 + parts[i].len() <= lim; let next = i + 1 < parts.len() && parts[i + 1].len() + 1 + parts[i].len() <= lim;
            if prev && (!next || parts[i - 1].len() <= parts[i + 1].len()) { let p = parts.remove(i); parts[i - 1].push(' '); parts[i - 1].extend(p); }
            else if next { let p = parts.remove(i + 1); parts[i].push(' '); parts[i].extend(p); } else { continue }
            changed = true; break; } }
    parts.into_iter().filter(|p| !p.is_empty()).map(|mut p| { if !".!?,;:\"\u{2019}\u{201d}')]".contains(*p.last().unwrap()) { p.push(',') } c2s(&p) }).collect()
}
pub fn characters(s: &str) -> usize { s.chars().count() }
pub fn join(chunks: &[Vec<f32>], gap: f32) -> Vec<f32> {
    let waves: Vec<&Vec<f32>> = chunks.iter().filter(|w| !w.is_empty()).collect();
    if waves.len() <= 1 { return waves.first().map(|w| (*w).clone()).unwrap_or_default() }
    let mut out = vec![];
    for (i, w) in waves.iter().enumerate() {
        let (mut start, mut end) = (0, w.len());
        if w.len() >= 1200 { let (mut first, mut last) = (w.len(), 0); let th = 10f32.powf(-45.0 / 20.0);
            for f in 0..w.len() / 240 { let sum: f32 = w[f * 240..f * 240 + 240].iter().map(|x| x * x).sum(); if (sum / 240.0 + 1e-12).sqrt() > th { first = first.min(f * 240); last = (f + 1) * 240 } }
            if first != w.len() { start = first.saturating_sub(480); end = w.len().min(last + 480) } }
        let mut part = w[start..end].to_vec(); let fade = 192;
        if part.len() > 2 * fade { let n = part.len(); for j in 0..fade { part[j] *= j as f32 / (fade - 1) as f32; part[n - fade + j] *= 1.0 - j as f32 / (fade - 1) as f32 } }
        out.extend(part); if i + 1 < waves.len() { out.extend(std::iter::repeat(0f32).take((24000.0 * gap) as usize)) }
    }
    out
}
