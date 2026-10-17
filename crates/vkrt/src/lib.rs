//! Thin Vulkan compute layer over ash, tuned for UMA (Strix Halo / RADV).
use anyhow::{anyhow, bail, Result};
use ash::vk;
use std::ffi::CStr;

pub mod shaders {
    pub const GEMV_Q4K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4k.spv"));
    pub const GEMV2_Q4K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv2_q4k.spv"));
    pub const GEMV2_Q6K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv2_q6k.spv"));
    pub const GEMV2_Q4K_SWIGLU: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv2_q4k_swiglu.spv"));
    pub const RMSNORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
    pub const ROPE_KV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_kv.spv"));
    pub fn gemv3(lpr: u32, swiglu: bool) -> &'static [u8] {
        match (lpr, swiglu) {
            (4, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_l4.spv")), (4, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_swiglu_l4.spv")),
            (8, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_l8.spv")), (8, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_swiglu_l8.spv")),
            (16, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_l16.spv")), (16, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_swiglu_l16.spv")),
            (_, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3.spv")), (_, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_swiglu.spv")),
        }
    }
    pub const GEMV3: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv3.spv"));
    pub const GEMV3_SWIGLU: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv3_swiglu.spv"));
    pub const EMBED_Q4K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed_q4k.spv"));
    pub const ROPE_KV16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_kv16.spv"));
    pub const ATTN16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn16.spv"));
    pub const ARGMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
    pub const SAMPLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample.spv"));
    pub const ARGMAX_W: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_w.spv"));
    pub const SAMPLE_W: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_w.spv"));
    pub const ROUTER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/router.spv"));
    pub const SHORTCONV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shortconv.spv"));
    pub const ROPE_QK64: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_qk64.spv"));
    pub const ATTN64: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn64.spv"));
    pub const EMBED_Q6K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed_q6k.spv"));
    pub const Q8K_ROUND: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q8k_round.spv"));
    pub const ATTN_SPLIT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_split.spv"));
    pub const ATTN_COMBINE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_combine.spv"));
    /// Generic-format GEMV (shaders/gemvq.comp + q_any.glsl), lanes-per-row variant.
    pub fn gemvq(lpr: u32, swiglu: bool) -> &'static [u8] {
        match (lpr, swiglu) {
            (4, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_l4.spv")), (4, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_swiglu_l4.spv")),
            (8, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_l8.spv")), (8, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_swiglu_l8.spv")),
            (16, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_l16.spv")), (16, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_swiglu_l16.spv")),
            (_, false) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_l32.spv")), (_, true) => include_bytes!(concat!(env!("OUT_DIR"), "/gemvq_swiglu_l32.spv")),
        }
    }
    pub const EMBED_Q4K_XW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed_q4k_xw.spv"));
    pub const GDN_CONV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_conv.spv"));
    pub const GDN_STEP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_step.spv"));
    pub const GDN_STEP_M: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_step_m.spv"));
    pub const GDN_CONV_M: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_conv_m.spv"));
    pub const ROPE_QKN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_qkn.spv"));
    pub const ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn.spv"));
    pub const EMBED_ANY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed_any.spv"));
    pub const RMSPREP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsprep.spv"));
    pub const ARGMAX_B: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_b.spv"));
    pub const XQ8: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/xq8.spv"));
    pub const MTP_PREP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mtp_prep.spv"));
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String, pub driver_name: String, pub driver_info: String, pub api_version: String,
    pub subgroup_size: u32, pub min_subgroup: u32, pub max_subgroup: u32,
    pub max_wg_count: [u32; 3], pub max_wg_invocations: u32, pub max_storage_buffer_range: u32, pub max_shared_mem: u32,
    pub coop_matrix: Vec<String>, pub heaps: Vec<String>, pub mem_types: Vec<String>, pub ts_period_ns: f32,
}

pub struct Ctx {
    pub entry: ash::Entry, pub instance: ash::Instance, pub pdev: vk::PhysicalDevice, pub device: ash::Device,
    pub queue: vk::Queue, pub qfam: u32, pub mem_props: vk::PhysicalDeviceMemoryProperties, pub info: DeviceInfo,
    pub has_coop_matrix: bool, pub has_host_import: bool, pub has_dmabuf: bool, cmd_pool: vk::CommandPool,
}

pub struct Buffer { pub buf: vk::Buffer, pub mem: vk::DeviceMemory, pub size: u64, pub ptr: *mut u8, pub flags: vk::MemoryPropertyFlags }
impl Buffer {
    pub fn bytes_mut(&self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) } }
    pub fn as_f32(&self) -> &[f32] { unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.size as usize / 4) } }
    pub fn as_f32_mut(&self) -> &mut [f32] { unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.size as usize / 4) } }
}

pub struct Kernel { pub pipeline: vk::Pipeline, pub layout: vk::PipelineLayout, pub dsl: vk::DescriptorSetLayout, pub pool: vk::DescriptorPool, pub n_bindings: u32 }

fn mpf(f: vk::MemoryPropertyFlags) -> String {
    let mut s = vec![];
    for (b, n) in [(vk::MemoryPropertyFlags::DEVICE_LOCAL, "DL"), (vk::MemoryPropertyFlags::HOST_VISIBLE, "HV"), (vk::MemoryPropertyFlags::HOST_COHERENT, "HC"), (vk::MemoryPropertyFlags::HOST_CACHED, "HCached")] { if f.contains(b) { s.push(n) } }
    s.join("|")
}

impl Ctx {
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default().application_name(c"strix-rt").api_version(vk::make_api_version(0, 1, 3, 0));
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)? };
        let pdevs = unsafe { instance.enumerate_physical_devices()? };
        let mut chosen = None;
        for &pd in &pdevs {
            let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p12);
            unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
            if p12.driver_id == vk::DriverId::MESA_RADV { chosen = Some(pd); break; }
        }
        let pdev = chosen.or(pdevs.first().copied()).ok_or_else(|| anyhow!("no vulkan device"))?;
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11).push_next(&mut p12).push_next(&mut p13);
        unsafe { instance.get_physical_device_properties2(pdev, &mut p2) };
        let props = p2.properties;
        let exts = unsafe { instance.enumerate_device_extension_properties(pdev)? };
        let has_ext = |n: &CStr| exts.iter().any(|e| e.extension_name_as_c_str().map(|x| x == n).unwrap_or(false));
        let has_coop_matrix = has_ext(ash::khr::cooperative_matrix::NAME);
        let mut coop = vec![];
        if has_coop_matrix {
            let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
            for p in unsafe { cm.get_physical_device_cooperative_matrix_properties(pdev)? } {
                coop.push(format!("M{}xN{}xK{} A={:?} B={:?} C={:?} R={:?} scope={:?}", p.m_size, p.n_size, p.k_size, p.a_type, p.b_type, p.c_type, p.result_type, p.scope));
            }
        }
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
        let heaps = (0..mem_props.memory_heap_count as usize).map(|i| { let h = mem_props.memory_heaps[i]; format!("heap{i}: {:.1} GiB flags={:?}", h.size as f64 / (1u64 << 30) as f64, h.flags) }).collect();
        let mem_types = (0..mem_props.memory_type_count as usize).map(|i| { let t = mem_props.memory_types[i]; format!("type{i}: heap{} {}", t.heap_index, mpf(t.property_flags)) }).collect();
        let info = DeviceInfo {
            name: props.device_name_as_c_str()?.to_string_lossy().into(),
            driver_name: p12.driver_name_as_c_str()?.to_string_lossy().into(),
            driver_info: p12.driver_info_as_c_str()?.to_string_lossy().into(),
            api_version: format!("{}.{}.{}", vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version), vk::api_version_patch(props.api_version)),
            subgroup_size: p11.subgroup_size, min_subgroup: p13.min_subgroup_size, max_subgroup: p13.max_subgroup_size,
            max_wg_count: props.limits.max_compute_work_group_count, max_wg_invocations: props.limits.max_compute_work_group_invocations,
            max_storage_buffer_range: props.limits.max_storage_buffer_range, max_shared_mem: props.limits.max_compute_shared_memory_size,
            coop_matrix: coop, heaps, mem_types, ts_period_ns: props.limits.timestamp_period,
        };
        let qfam = unsafe { instance.get_physical_device_queue_family_properties(pdev) }.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).ok_or_else(|| anyhow!("no compute queue"))? as u32;
        let qp = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfam).queue_priorities(&qp)];
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().shader_float16(true).shader_int8(true).storage_buffer8_bit_access(true).buffer_device_address(true).shader_buffer_int64_atomics(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true).compute_full_subgroups(true);
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut dev_exts = vec![];
        if has_coop_matrix { dev_exts.push(ash::khr::cooperative_matrix::NAME.as_ptr()); }
        let has_host_import = has_ext(ash::ext::external_memory_host::NAME);
        if has_host_import { dev_exts.push(ash::ext::external_memory_host::NAME.as_ptr()); }
        let has_dmabuf = has_ext(ash::ext::external_memory_dma_buf::NAME) && has_ext(ash::khr::external_memory_fd::NAME);
        if has_dmabuf { dev_exts.push(ash::ext::external_memory_dma_buf::NAME.as_ptr()); dev_exts.push(ash::khr::external_memory_fd::NAME.as_ptr()); }
        let has_af = has_ext(ash::ext::shader_atomic_float::NAME);
        let has_cr = has_ext(ash::ext::conditional_rendering::NAME);
        if has_cr { dev_exts.push(ash::ext::conditional_rendering::NAME.as_ptr()); HAS_COND.store(true, std::sync::atomic::Ordering::Relaxed); }
        let mut fcr = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default().conditional_rendering(true);
        if has_af { dev_exts.push(ash::ext::shader_atomic_float::NAME.as_ptr()); }
        let mut faf = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default().shader_buffer_float32_atomic_add(true);
        let base_feats = vk::PhysicalDeviceFeatures::default().shader_int64(true).shader_int16(true);
        let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&dev_exts).enabled_features(&base_feats).push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
        if has_af { dci = dci.push_next(&mut faf); }
        if has_cr { dci = dci.push_next(&mut fcr); }
        if has_coop_matrix { dci = dci.push_next(&mut fcm); }
        let device = unsafe { instance.create_device(pdev, &dci, None)? };
        let queue = unsafe { device.get_device_queue(qfam, 0) };
        let cmd_pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(qfam).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)? };
        Ok(Ctx { entry, instance, pdev, device, queue, qfam, mem_props, info, has_coop_matrix, has_host_import, has_dmabuf, cmd_pool })
    }

    /// Allocate a persistently mapped storage buffer. Prefers HOST_VISIBLE|DEVICE_LOCAL (UMA zero-copy).
    pub fn buffer(&self, size: u64) -> Result<Buffer> { self.buffer_pref(size, false) }
    /// Host-cached system memory: for buffers the CPU reads back (logits); UMA DL|HV memory is write-combined/uncached for CPU reads.
    pub fn buffer_readback(&self, size: u64) -> Result<Buffer> { self.buffer_pref(size, true) }
    fn buffer_pref(&self, size: u64, cached: bool) -> Result<Buffer> {
        let d = &self.device;
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        if HAS_COND.load(std::sync::atomic::Ordering::Relaxed) { usage |= vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT; }
        let bci = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buf = unsafe { d.create_buffer(&bci, None)? };
        let req = unsafe { d.get_buffer_memory_requirements(buf) };
        let hcc = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED;
        let want_c = [hcc, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT];
        let want_d = [vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT];
        let mut idx = None;
        let want = if cached { want_c } else { want_d };
        'o: for w in want { for i in 0..self.mem_props.memory_type_count { let t = self.mem_props.memory_types[i as usize];
            if req.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(w) && self.mem_props.memory_heaps[t.heap_index as usize].size >= req.size { idx = Some(i); break 'o; } } }
        let idx = idx.ok_or_else(|| anyhow!("no host-visible memory type"))?;
        let mem = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx), None)? };
        unsafe { d.bind_buffer_memory(buf, mem, 0)? };
        let ptr = unsafe { d.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? } as *mut u8;
        Ok(Buffer { buf, mem, size, ptr, flags: self.mem_props.memory_types[idx as usize].property_flags })
    }

    pub fn kernel(&self, spirv: &[u8], n_bindings: u32, push_size: u32, spec: &[u32], subgroup: Option<u32>) -> Result<Kernel> {
        let d = &self.device;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))?;
        let sm = unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? };
        let b: Vec<_> = (0..n_bindings).map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let dsl = unsafe { d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&b), None)? };
        let pcr = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(push_size)];
        let dsls = [dsl];
        let layout = unsafe { d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None)? };
        let entries: Vec<_> = (0..spec.len() as u32).map(|i| vk::SpecializationMapEntry::default().constant_id(i).offset(i * 4).size(4)).collect();
        let data: Vec<u8> = spec.iter().flat_map(|v| v.to_le_bytes()).collect();
        let si = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let mut rss = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(subgroup.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(sm).name(c"main").specialization_info(&si);
        if subgroup.is_some() { stage = stage.push_next(&mut rss).flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS); }
        // DISPATCH_BASE: dispatch() splits > 65535 workgroups into 1-D chunks with a base offset (see dispatch)
        let ci = [vk::ComputePipelineCreateInfo::default().flags(vk::PipelineCreateFlags::DISPATCH_BASE).stage(stage).layout(layout)];
        let pipeline = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &ci, None).map_err(|(_, e)| e)?[0] };
        unsafe { d.destroy_shader_module(sm, None) };
        let ps = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1024 * n_bindings.max(1))];
        let pool = unsafe { d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1024).pool_sizes(&ps), None)? };
        Ok(Kernel { pipeline, layout, dsl, pool, n_bindings })
    }

    pub fn bind(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> {
        if bufs.len() as u32 != k.n_bindings { bail!("binding count mismatch") }
        let dsls = [k.dsl];
        let ds = unsafe { self.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(k.pool).set_layouts(&dsls))?[0] };
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter().map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buf).offset(0).range(vk::WHOLE_SIZE)]).collect();
        let w: Vec<_> = infos.iter().enumerate().map(|(i, inf)| vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(inf)).collect();
        unsafe { self.device.update_descriptor_sets(&w, &[]) };
        Ok(ds)
    }

    /// Bind + push + dispatch `groups` workgroups (split into 2D when > 65535).
    pub fn dispatch(&self, cb: vk::CommandBuffer, k: &Kernel, ds: vk::DescriptorSet, push: &[u8], groups: u32) {
        let d = &self.device;
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, k.layout, 0, &[ds], &[]);
            if !push.is_empty() { d.cmd_push_constants(cb, k.layout, vk::ShaderStageFlags::COMPUTE, 0, push); }
            // Exactly `groups` workgroups, 1-D: chunks of <= 65535 via vkCmdDispatchBase, so gl_WorkGroupID.x is the global
            // linear id (y = 0). The old (gx, gy) split launched gx*gy >= groups workgroups and broke shaders that only
            // read gl_WorkGroupID.x (e.g. in-place pf_rope64 rotated rows twice once n*(heads) > 65535: lfm2 L=2048).
            let mut b = 0u32;
            while b < groups { let c = (groups - b).min(65535); d.cmd_dispatch_base(cb, b, 0, 0, c, 1, 1); b += c; }
        }
    }
    /// 3-D dispatch (x, y, z workgroups).
    pub fn dispatch3(&self, cb: vk::CommandBuffer, k: &Kernel, ds: vk::DescriptorSet, push: &[u8], x: u32, y: u32, z: u32) {
        let d = &self.device;
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, k.layout, 0, &[ds], &[]);
            if !push.is_empty() { d.cmd_push_constants(cb, k.layout, vk::ShaderStageFlags::COMPUTE, 0, push); }
            d.cmd_dispatch(cb, x, y, z);
        }
    }
    pub fn barrier(&self, cb: vk::CommandBuffer) {
        let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        unsafe { self.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &mb, &[], &[]) };
    }

    /// Record with `f`, submit, wait. Returns GPU time in seconds between first and last command (timestamps).
    pub fn run<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<f64> {
        let d = &self.device;
        unsafe {
            let cb = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            let qp = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            d.cmd_reset_query_pool(cb, qp, 0, 2);
            d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            f(cb);
            d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
            d.end_command_buffer(cb)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [cb];
            // Many back-to-back submits of <= CHUNK command buffers, fence only on the last: the host never waits in between.
            // (One vkQueueSubmit with ~130 copies of a SIMULTANEOUS_USE buffer hung the GPU on RADV 26.0.8.)
            let chunk: usize = std::env::var("SUBMIT_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
            let parts: Vec<&[vk::CommandBuffer]> = cbs.chunks(chunk).collect();
            for (i, p) in parts.iter().enumerate() {
                d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(p)], if i + 1 == parts.len() { fence } else { vk::Fence::null() })?;
            }
            d.wait_for_fences(&[fence], true, u64::MAX)?;
            let mut ts = [0u64; 2];
            d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
            d.destroy_fence(fence, None); d.destroy_query_pool(qp, None); d.free_command_buffers(self.cmd_pool, &cbs);
            Ok((ts[1].wrapping_sub(ts[0])) as f64 * self.info.ts_period_ns as f64 * 1e-9)
        }
    }
}

pub fn push_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// A command buffer recorded once and resubmitted many times (whole-token graph).
pub struct Recorded { pub cb: vk::CommandBuffer, qp: vk::QueryPool, fence: vk::Fence }
impl Ctx {
    pub fn record<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<Recorded> {
        let d = &self.device;
        unsafe {
            let cb = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            let qp = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(cb, qp, 0, 2);
            d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            f(cb);
            d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
            d.end_command_buffer(cb)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(Recorded { cb, qp, fence })
        }
    }
    /// Submit, wait; returns GPU seconds.
    pub fn submit(&self, r: &Recorded) -> Result<f64> {
        let d = &self.device;
        unsafe {
            d.reset_fences(&[r.fence])?;
            let cbs = [r.cb];
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], r.fence)?;
            d.wait_for_fences(&[r.fence], true, u64::MAX)?;
            let mut ts = [0u64; 2];
            d.get_query_pool_results(r.qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
            Ok((ts[1].wrapping_sub(ts[0])) as f64 * self.info.ts_period_ns as f64 * 1e-9)
        }
    }
}

impl Ctx {
    /// Zero-copy: wrap existing host memory (e.g. a file-backed mmap) as a storage buffer via VK_EXT_external_memory_host.
    /// `ptr` and `size` must be multiples of minImportedHostPointerAlignment (4 KiB on RADV). The memory must outlive the buffer.
    pub unsafe fn import_host(&self, ptr: *mut u8, size: u64) -> Result<Buffer> {
        if !self.has_host_import { bail!("VK_EXT_external_memory_host not supported") }
        let d = &self.device;
        let ext = ash::ext::external_memory_host::Device::new(&self.instance, d);
        let ht = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut hp = vk::MemoryHostPointerPropertiesEXT::default();
        (ext.fp().get_memory_host_pointer_properties_ext)(d.handle(), ht, ptr as *const std::ffi::c_void, &mut hp).result().map_err(|e| anyhow!("get_memory_host_pointer_properties: {e:?}"))?;
        let mut emb = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
        let bci = vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS).push_next(&mut emb);
        let buf = d.create_buffer(&bci, None)?;
        let req = d.get_buffer_memory_requirements(buf);
        let bits = req.memory_type_bits & hp.memory_type_bits;
        // prefer DEVICE_LOCAL among importable types
        let mut idx = None;
        for pref in [true, false] { for i in 0..self.mem_props.memory_type_count { if bits & (1 << i) != 0 && (!pref || self.mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)) { idx.get_or_insert(i); } } if idx.is_some() { break } }
        let idx = idx.ok_or_else(|| anyhow!("no importable memory type (host ptr bits {:#x}, buffer bits {:#x})", hp.memory_type_bits, req.memory_type_bits))?;
        let mut imp = vk::ImportMemoryHostPointerInfoEXT::default().handle_type(ht).host_pointer(ptr as *mut std::ffi::c_void);
        let mut af = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mem = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(idx).push_next(&mut imp).push_next(&mut af), None).map_err(|e| anyhow!("allocate_memory(import host ptr): {e:?}"))?;
        d.bind_buffer_memory(buf, mem, 0)?;
        Ok(Buffer { buf, mem, size, ptr, flags: self.mem_props.memory_types[idx as usize].property_flags })
    }
    pub fn free_buffer(&self, b: Buffer) { unsafe { self.device.destroy_buffer(b.buf, None); self.device.free_memory(b.mem, None); } }
}

impl Ctx {
    /// Import a dma-buf fd (ownership passes to Vulkan on success). Prefers non-host-visible DEVICE_LOCAL types.
    pub unsafe fn import_dmabuf(&self, fd: i32, size: u64) -> Result<Buffer> {
        if !self.has_dmabuf { bail!("dma-buf import not supported") }
        let d = &self.device;
        let ht = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let efd = ash::khr::external_memory_fd::Device::new(&self.instance, d);
        let mut fp = vk::MemoryFdPropertiesKHR::default();
        (efd.fp().get_memory_fd_properties_khr)(d.handle(), ht, fd, &mut fp).result().map_err(|e| anyhow!("get_memory_fd_properties: {e:?}"))?;
        let mut emb = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
        let buf = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS).push_next(&mut emb), None)?;
        let req = d.get_buffer_memory_requirements(buf);
        let bits = req.memory_type_bits & fp.memory_type_bits;
        let pref = std::env::var("DMABUF_TYPE").ok().and_then(|v| v.parse::<u32>().ok());
        let mut idx = pref.filter(|i| bits & (1 << i) != 0);
        for want_dl in [true, false] { if idx.is_some() { break } for i in 0..self.mem_props.memory_type_count { let f = self.mem_props.memory_types[i as usize].property_flags;
            if bits & (1 << i) != 0 && (!want_dl || f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)) { idx = Some(i); break } } }
        let idx = idx.ok_or_else(|| anyhow!("no memory type for dma-buf (fd bits {:#x}, buf bits {:#x})", fp.memory_type_bits, req.memory_type_bits))?;
        let mut imp = vk::ImportMemoryFdInfoKHR::default().handle_type(ht).fd(fd);
        let mut ded = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
        let mut af = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mem = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size.max(size)).memory_type_index(idx).push_next(&mut imp).push_next(&mut ded).push_next(&mut af), None).map_err(|e| anyhow!("allocate_memory(dma-buf): {e:?}"))?;
        d.bind_buffer_memory(buf, mem, 0)?;
        Ok(Buffer { buf, mem, size, ptr: std::ptr::null_mut(), flags: self.mem_props.memory_types[idx as usize].property_flags })
    }
    /// Memory type bits a dma-buf fd can be imported as (for reporting).
    pub fn dmabuf_type_bits(&self, fd: i32) -> u32 {
        let efd = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);
        let mut fp = vk::MemoryFdPropertiesKHR::default();
        unsafe { let _ = (efd.fp().get_memory_fd_properties_khr)(self.device.handle(), vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT, fd, &mut fp); }
        fp.memory_type_bits
    }
}

/// /dev/strixpc (kmod/strixpc): pin page-cache pages behind a user mapping and export them as a dma-buf fd.
pub mod pagecache {
    use anyhow::{bail, Result};
    #[repr(C)] struct Export { addr: u64, len: u64, fd: i32, flags: u32 }
    const IOC_EXPORT: libc::c_ulong = (3 << 30) | ((std::mem::size_of::<Export>() as libc::c_ulong) << 16) | ((b'S' as libc::c_ulong) << 8) | 1;
    pub fn available() -> bool { std::path::Path::new("/dev/strixpc").exists() }
    /// `addr`/`len` page aligned. Returns a dma-buf fd. Pages stay pinned until the dma-buf (and all importers) are released.
    pub unsafe fn export(addr: *const u8, len: usize) -> Result<i32> {
        let dev = libc::open(c"/dev/strixpc".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
        if dev < 0 { bail!("open /dev/strixpc: {}", std::io::Error::last_os_error()) }
        let mut e = Export { addr: addr as u64, len: len as u64, fd: -1, flags: 0 };
        let r = libc::ioctl(dev, IOC_EXPORT, &mut e as *mut Export);
        let err = std::io::Error::last_os_error(); libc::close(dev);
        if r != 0 { bail!("STRIXPC_IOC_EXPORT: {err}") }
        Ok(e.fd)
    }
}

impl Ctx { pub fn device_address(&self, b: &Buffer) -> u64 { unsafe { self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(b.buf)) } } }

impl Ctx {
    /// Record a reusable command buffer (SIMULTANEOUS_USE so the same buffer can appear many times in one batch).
    pub fn record_raw<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<vk::CommandBuffer> {
        let d = &self.device;
        unsafe {
            let cb = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE))?;
            f(cb);
            d.end_command_buffer(cb)?;
            Ok(cb)
        }
    }
    /// Submit `body` repeated `n` times in ONE vkQueueSubmit, bracketed by GPU timestamps; host waits once at the end.
    /// Returns GPU seconds for the whole batch.
    pub fn submit_repeated(&self, body: vk::CommandBuffer, n: usize) -> Result<f64> {
        let d = &self.device;
        unsafe {
            let qp = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)?;
            let t0 = self.record_raw(|cb| { d.cmd_reset_query_pool(cb, qp, 0, 2); d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0); })?;
            let t1 = self.record_raw(|cb| { d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1); })?;
            let mut cbs = vec![t0]; cbs.extend(std::iter::repeat(body).take(n)); cbs.push(t1);
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            // Many back-to-back submits of <= CHUNK command buffers, fence only on the last: the host never waits in between.
            // (One vkQueueSubmit with ~130 copies of a SIMULTANEOUS_USE buffer hung the GPU on RADV 26.0.8.)
            let chunk: usize = std::env::var("SUBMIT_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
            let parts: Vec<&[vk::CommandBuffer]> = cbs.chunks(chunk).collect();
            for (i, p) in parts.iter().enumerate() {
                d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(p)], if i + 1 == parts.len() { fence } else { vk::Fence::null() })?;
            }
            d.wait_for_fences(&[fence], true, u64::MAX)?;
            let mut ts = [0u64; 2];
            d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
            d.destroy_fence(fence, None); d.destroy_query_pool(qp, None); d.free_command_buffers(self.cmd_pool, &[t0, t1]);
            Ok(ts[1].wrapping_sub(ts[0]) as f64 * self.info.ts_period_ns as f64 * 1e-9)
        }
    }
}

/// VK_EXT_conditional_rendering enabled on the device (buffers then also carry CONDITIONAL_RENDERING usage)
pub static HAS_COND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Experimental weight placement (opt-in, see qimage st.rs QI_WMEM): a storage buffer in an explicit memory type index.
/// Non-host-visible types get ptr = null; fill them with `upload` (staged copies).
impl Ctx {
    pub fn buffer_in_type(&self, size: u64, type_idx: u32) -> Result<Buffer> {
        let d = &self.device;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buf = unsafe { d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)? };
        let req = unsafe { d.get_buffer_memory_requirements(buf) };
        if req.memory_type_bits & (1 << type_idx) == 0 { bail!("memory type {type_idx} not allowed for buffers (bits {:#x})", req.memory_type_bits) }
        let mem = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(type_idx), None)? };
        unsafe { d.bind_buffer_memory(buf, mem, 0)? };
        let flags = self.mem_props.memory_types[type_idx as usize].property_flags;
        let ptr = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) { (unsafe { d.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? }) as *mut u8 } else { std::ptr::null_mut() };
        Ok(Buffer { buf, mem, size, ptr, flags })
    }
    /// Copy host bytes into `dst` at byte offset 0 through a 256 MB host-visible staging buffer.
    pub fn upload(&self, dst: &Buffer, src: &[u8]) -> Result<()> {
        if !dst.ptr.is_null() { dst.bytes_mut()[..src.len()].copy_from_slice(src); return Ok(()) }
        const CH: usize = 256 << 20; let st = self.buffer(CH.min(src.len()).max(4) as u64)?;
        for (i, c) in src.chunks(CH).enumerate() {
            st.bytes_mut()[..c.len()].copy_from_slice(c);
            let region = vk::BufferCopy::default().src_offset(0).dst_offset((i * CH) as u64).size(c.len() as u64);
            self.run(|cb| unsafe { self.device.cmd_copy_buffer(cb, st.buf, dst.buf, &[region]) })?;
        }
        self.free_buffer(st); Ok(())
    }
}
