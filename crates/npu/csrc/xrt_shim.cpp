// Minimal extern "C" shim over XRT (libxrt_coreutil) for the amdxdna NPU. Every call catches C++ exceptions and
// stores the message for npu_last_error().
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <map>
#include <memory>
#include <cstdlib>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <cstring>
#include <unistd.h>
#include <string>
#include <vector>

static thread_local std::string g_err;
struct NpuCtx {
  std::shared_ptr<struct NpuShared> sh; xrt::device dev; xrt::kernel k;
  xrt::bo instr, tmp, trace; uint32_t ninstr = 0;
};
// One device + hw_context + kernel per xclbin path, shared by every npu_open of that path: switching between
// hardware contexts costs ~2 ms per run on XDNA2, so shapes whose instruction streams run on the same xclbin
// (env NPU_XCLBIN overrides the path for all opens) share one context.
struct NpuShared { xrt::device dev; xrt::xclbin xb; xrt::hw_context hw; xrt::kernel k; };
static std::map<std::string, std::weak_ptr<NpuShared>> g_shared;
struct NpuBo { xrt::bo bo; };

#define TRY(x) try { x } catch (const std::exception& e) { g_err = e.what(); } catch (...) { g_err = "unknown C++ exception"; }

extern "C" {
const char* npu_last_error() { return g_err.c_str(); }

void* npu_open(unsigned dev_idx, const char* xclbin, const char* kname, const uint32_t* instr, size_t ninstr) {
  TRY({
    auto* c = new NpuCtx();
    const char* ov = getenv("NPU_XCLBIN"); std::string path = (ov && *ov) ? ov : xclbin;
    std::string key = path + "|" + kname;
    auto sh = g_shared[key].lock();
    if (!sh) {
      sh = std::make_shared<NpuShared>();
      sh->dev = xrt::device(dev_idx);
      sh->xb = xrt::xclbin(path);
      sh->dev.register_xclbin(sh->xb);
      sh->hw = xrt::hw_context(sh->dev, sh->xb.get_uuid());
      sh->k = xrt::kernel(sh->hw, kname);
      g_shared[key] = sh;
    }
    c->sh = sh; c->dev = sh->dev; c->k = sh->k;
    c->instr = xrt::bo(c->dev, ninstr * 4, XCL_BO_FLAGS_CACHEABLE, c->k.group_id(1));
    std::memcpy(c->instr.map<void*>(), instr, ninstr * 4);
    c->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c->ninstr = (uint32_t)ninstr;
    c->tmp = xrt::bo(c->dev, 1, XRT_BO_FLAGS_HOST_ONLY, c->k.group_id(6));
    c->trace = xrt::bo(c->dev, 4, XRT_BO_FLAGS_HOST_ONLY, c->k.group_id(7));
    return (void*)c;
  })
  return nullptr;
}
void npu_close(void* c) { TRY({ delete (NpuCtx*)c; }) }

void* npu_bo_alloc(void* cv, size_t sz, int arg) {
  auto* c = (NpuCtx*)cv;
  TRY({ return (void*)new NpuBo{xrt::bo(c->dev, sz, XRT_BO_FLAGS_HOST_ONLY, c->k.group_id(arg))}; })
  return nullptr;
}
void* npu_bo_userptr(void* cv, void* p, size_t sz, int arg) {
  auto* c = (NpuCtx*)cv;
  TRY({ return (void*)new NpuBo{xrt::bo(c->dev, p, sz, XRT_BO_FLAGS_HOST_ONLY, c->k.group_id(arg))}; })
  return nullptr;
}
void* npu_bo_import_fd(void* cv, int fd) {
  auto* c = (NpuCtx*)cv;
  TRY({ return (void*)new NpuBo{xrt::bo(c->dev, (xrt::bo::export_handle)fd)}; })
  return nullptr;
}
int npu_bo_export_fd(void* b) { TRY({ return (int)((NpuBo*)b)->bo.export_buffer(); }) return -1; }
void* npu_bo_map(void* b) { TRY({ return ((NpuBo*)b)->bo.map<void*>(); }) return nullptr; }
size_t npu_bo_size(void* b) { TRY({ return ((NpuBo*)b)->bo.size(); }) return 0; }
uint64_t npu_bo_address(void* b) { TRY({ return ((NpuBo*)b)->bo.address(); }) return 0; }
int npu_bo_sync(void* b, int to_device) {
  TRY({ ((NpuBo*)b)->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE); return 0; })
  return -1;
}
void npu_bo_free(void* b) { TRY({ delete (NpuBo*)b; }) }

// Runs the loaded instruction stream with (a, b, c) on kernel args 3, 4, 5. Returns the ert_cmd_state
// (4 = COMPLETED) or -1 on exception.
int npu_run(void* cv, void* a, void* b, void* out) {
  auto* c = (NpuCtx*)cv;
  TRY({
    auto run = c->k(3u, c->instr, c->ninstr, ((NpuBo*)a)->bo, ((NpuBo*)b)->bo, ((NpuBo*)out)->bo, c->tmp, c->trace);
    return (int)run.wait();
  })
  return -1;
}
}
