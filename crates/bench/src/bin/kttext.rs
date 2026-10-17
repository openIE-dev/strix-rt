//! `kttext` : JSON lines {"text": ...} on stdin -> {"normalized", "chunks", "expression"} (Rust port of the fork's --frontend, backend off)
use std::io::BufRead;
fn q(s: &str) -> String { let mut o = String::from("\""); for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } } o.push('"'); o }
fn main() -> anyhow::Result<()> {
    for line in std::io::stdin().lock().lines() { let line = line?; if line.trim().is_empty() { continue }
        let j = kitten::json::parse(&line)?; let t = j.at("text")?.str().to_string();
        let n = kitten::text::normalize(&t); let ch = kitten::text::split(&n, 380, 130);
        println!("{{\"normalized\":{},\"chunks\":[{}],\"expression\":{}}}", q(&n), ch.iter().map(|c| q(c)).collect::<Vec<_>>().join(","), kitten::text::has_expression(&t)); }
    Ok(())
}
