//! Minimal PNG writer (8-bit RGB/RGBA, zlib stored blocks).
fn crc32(d: &[u8], mut c: u32) -> u32 { c = !c; for &b in d { c ^= b as u32; for _ in 0..8 { c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 } } } !c }
pub fn write(path: &str, w: u32, h: u32, ch: u32, px: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((h * (w * ch + 1)) as usize);
    for y in 0..h as usize { raw.push(0); raw.extend_from_slice(&px[y * (w * ch) as usize..(y + 1) * (w * ch) as usize]); }
    let mut z = vec![0x78, 0x01]; let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw { a = (a + x as u32) % 65521; b = (b + a) % 65521; }
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() { z.push(if chunks.peek().is_none() { 1 } else { 0 }); let l = c.len() as u16; z.extend_from_slice(&l.to_le_bytes()); z.extend_from_slice(&(!l).to_le_bytes()); z.extend_from_slice(c); }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |ty: &[u8], data: &[u8]| { out.extend_from_slice(&(data.len() as u32).to_be_bytes()); let mut c = ty.to_vec(); c.extend_from_slice(data); out.extend_from_slice(&c); out.extend_from_slice(&crc32(&c, 0).to_be_bytes()); };
    let mut ih = vec![]; ih.extend_from_slice(&w.to_be_bytes()); ih.extend_from_slice(&h.to_be_bytes()); ih.extend_from_slice(&[8, if ch == 4 { 6 } else { 2 }, 0, 0, 0]);
    chunk(b"IHDR", &ih); chunk(b"IDAT", &z); chunk(b"IEND", &[]);
    std::fs::write(path, out)
}
