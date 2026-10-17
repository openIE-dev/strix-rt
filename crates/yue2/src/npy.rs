//! Minimal .npy (v1/v2, little-endian <f4 / <i4 / <i8) reader and <f4 writer.
use anyhow::{bail, Result};
pub fn read(path: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let b = std::fs::read(path)?; if b.len() < 10 || &b[..6] != b"\x93NUMPY" { bail!("{path}: not npy") }
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[off..off + hl])?;
    let sh = h.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = sh.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let d = &b[off + hl..];
    let v = if h.contains("'<f4'") { d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
        else if h.contains("'<i4'") { d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect() }
        else if h.contains("'<i8'") { d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect() }
        else { bail!("{path}: dtype {h}") };
    Ok((v, shape))
}
pub fn read_u32(path: &str) -> Result<Vec<u32>> { Ok(read(path)?.0.iter().map(|&x| x as u32).collect()) }
pub fn write(path: &str, v: &[f32], shape: &[usize]) -> Result<()> {
    let sh = shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ") + if shape.len() == 1 { "," } else { "" };
    let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({sh}), }}"); while (10 + h.len() + 1) % 64 != 0 { h.push(' ') } h.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec(); b.extend((h.len() as u16).to_le_bytes()); b.extend(h.as_bytes()); for x in v { b.extend(x.to_le_bytes()) }
    std::fs::write(path, b)?; Ok(())
}
/// Write interleaved f32 samples as a 32-bit float WAV.
pub fn write_wav(path: &str, s: &[f32], ch: u16, sr: u32) -> Result<()> {
    let n = (s.len() * 4) as u32; let mut b = vec![];
    b.extend(b"RIFF"); b.extend((36 + n).to_le_bytes()); b.extend(b"WAVEfmt "); b.extend(16u32.to_le_bytes()); b.extend(3u16.to_le_bytes()); b.extend(ch.to_le_bytes());
    b.extend(sr.to_le_bytes()); b.extend((sr * ch as u32 * 4).to_le_bytes()); b.extend((ch * 4).to_le_bytes()); b.extend(32u16.to_le_bytes()); b.extend(b"data"); b.extend(n.to_le_bytes());
    for x in s { b.extend(x.to_le_bytes()) } std::fs::write(path, b)?; Ok(())
}
