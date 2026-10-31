//! Print the Vulkan device info vkrt sees: limits, subgroים, every cooperative-matrix config.
fn main() -> anyhow::Result<()> {
    let ctx = vkrt::Ctx::new()?;
    let i = &ctx.info;
    println!("{} | driver {} {} | api {}", i.name, i.driver_name, i.driver_info, i.api_version);
    println!("subgroup {} (min {} max {}) | max shared {} B | max wg inv {}", i.subgroup_size, i.min_subgroup, i.max_subgroup, i.max_shared_mem, i.max_wg_invocations);
    for c in &i.coop_matrix { println!("coopmat {}", c); }
    Ok(())
}
