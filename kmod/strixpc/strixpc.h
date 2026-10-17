/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/* strixpc: export pinned page-cache (or any user) pages as a dma-buf. Shared uapi. */
#ifndef STRIXPC_H
#define STRIXPC_H
#include <linux/types.h>
#include <linux/ioctl.h>
struct strixpc_export {
	__u64 addr;   /* user VA of the range (page aligned), e.g. a read-only MAP_SHARED mmap of a GGUF file */
	__u64 len;    /* bytes, page multiple */
	__s32 fd;     /* out: dma-buf fd */
	__u32 flags;  /* reserved, 0 */
};
#define STRIXPC_IOC_EXPORT _IOWR('S', 1, struct strixpc_export)
#endif
