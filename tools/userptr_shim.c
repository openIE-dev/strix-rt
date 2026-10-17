// LD_PRELOAD shim: lets RADV import file-backed (page-cache) mmaps via VK_EXT_external_memory_host.
// libdrm's amdgpu_create_bo_from_user_mem always passes AMDGPU_GEM_USERPTR_ANONONLY, which the kernel
// rejects for file-backed VMAs (EPERM). We drop ANONONLY and add READONLY (GPU PTEs read-only; weights only).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
static int (*real_ioctl)(int, unsigned long, ...);
int ioctl(int fd, unsigned long req, ...) {
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    if (req == DRM_IOCTL_AMDGPU_GEM_USERPTR) {
        struct drm_amdgpu_gem_userptr *a = arg;
        unsigned old = a->flags;
        a->flags = (a->flags & ~AMDGPU_GEM_USERPTR_ANONONLY) | AMDGPU_GEM_USERPTR_READONLY;
        int r = real_ioctl(fd, req, arg);
        if (getenv("STRIX_SHIM_VERBOSE")) fprintf(stderr, "[shim] userptr addr=%#llx size=%llu flags %#x->%#x r=%d\n", (unsigned long long)a->addr, (unsigned long long)a->size, old, a->flags, r);
        return r;
    }
    return real_ioctl(fd, req, arg);
}
