#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
int main(int c, char **v) { int fd = open("/dev/dri/renderD128", O_RDWR), ff = open(v[1], O_RDONLY); size_t sz = 16ul << 20;
  void *p = mmap(0, sz, PROT_READ, MAP_SHARED, ff, 0);
  struct drm_amdgpu_gem_userptr a = { .addr = (unsigned long)p, .size = sz, .flags = AMDGPU_GEM_USERPTR_READONLY | AMDGPU_GEM_USERPTR_REGISTER | AMDGPU_GEM_USERPTR_VALIDATE };
  printf("userptr: %s\n", ioctl(fd, DRM_IOCTL_AMDGPU_GEM_USERPTR, &a) ? strerror(errno) : "OK");
  struct drm_prime_handle h = { .handle = a.handle, .flags = DRM_CLOEXEC | DRM_RDWR };
  printf("prime export of file-backed userptr BO: %s\n", ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &h) ? strerror(errno) : "OK"); return 0; }
