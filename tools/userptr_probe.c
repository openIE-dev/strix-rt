// Probe amdgpu DRM_IOCTL_AMDGPU_GEM_USERPTR on file-backed vs anonymous mmaps with various flags.
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
static int try(int fd, void *p, size_t sz, unsigned flags, const char *what) {
    struct drm_amdgpu_gem_userptr a = { .addr = (unsigned long)p, .size = sz, .flags = flags };
    int r = ioctl(fd, DRM_IOCTL_AMDGPU_GEM_USERPTR, &a);
    printf("%-28s flags=%#x -> %s\n", what, flags, r ? strerror(errno) : "OK");
    if (!r) { struct drm_gem_close c = { .handle = a.handle }; ioctl(fd, DRM_IOCTL_GEM_CLOSE, &c); }
    return r;
}
int main(int argc, char **argv) {
    int fd = open("/dev/dri/renderD128", O_RDWR); if (fd < 0) { perror("render"); return 1; }
    int ff = open(argv[1], O_RDONLY); size_t sz = 64ul << 20;
    void *fro = mmap(0, sz, PROT_READ, MAP_SHARED, ff, 0);
    void *fpw = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE, ff, 0);
    void *an = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); memset(an, 1, sz);
    volatile char s = 0; for (size_t i = 0; i < sz; i += 4096) s += ((char*)fro)[i] + ((char*)fpw)[i];
    unsigned A = AMDGPU_GEM_USERPTR_ANONONLY, R = AMDGPU_GEM_USERPTR_REGISTER, V = AMDGPU_GEM_USERPTR_VALIDATE, RO = AMDGPU_GEM_USERPTR_READONLY;
    struct { void *p; const char *n; } m[] = { { an, "anon rw" }, { fro, "file MAP_SHARED ro" }, { fpw, "file MAP_PRIVATE rw" } };
    unsigned fl[] = { A|R|V, R|V, RO|R|V, RO|R, RO };
    for (int i = 0; i < 3; i++) for (int j = 0; j < 5; j++) try(fd, m[i].p, sz, fl[j], m[i].n);
    return 0;
}
