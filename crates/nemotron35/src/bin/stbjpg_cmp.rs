//! `stbjpg_cmp <corpus dir> <ref dir>`: decode every JPEG with nemotron35::stbjpeg and compare byte for byte with the stb_image dump (logs/n35/jpgtest/stbdump.c).
fn main() {
    let a: Vec<String> = std::env::args().collect(); let (mut ok, mut bad, mut fail) = (0, 0, 0);
    let mut names: Vec<_> = std::fs::read_dir(&a[1]).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect(); names.sort();
    for n in names {
        let refp = format!("{}/{}.rgb", a[2], n); let rb = std::fs::read(&refp).ok();
        let bytes = std::fs::read(format!("{}/{}", a[1], n)).unwrap();
        match (nemotron35::stbjpeg::decode_rgb(&bytes), rb) {
            (Ok((px, w, h)), Some(r)) => { let (rw, rh) = (i32::from_le_bytes(r[0..4].try_into().unwrap()) as usize, i32::from_le_bytes(r[4..8].try_into().unwrap()) as usize);
                let nd = if rw == w && rh == h { px.iter().zip(&r[8..]).filter(|(x, y)| x != y).count() } else { usize::MAX };
                if nd == 0 { ok += 1; println!("EXACT {n} {w}x{h}") } else { bad += 1; println!("DIFF  {n} {w}x{h} vs {rw}x{rh}: {nd} bytes differ") } }
            (Err(e), Some(_)) => { fail += 1; println!("ERR   {n}: {e}") }
            (Err(_), None) => { ok += 1; println!("both fail {n}") }
            (Ok(_), None) => { bad += 1; println!("DIFF  {n}: we decode, stb fails") }
        }
    }
    println!("TOTAL exact {ok}, differ {bad}, our errors {fail}");
}
