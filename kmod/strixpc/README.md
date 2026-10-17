# strixpc (out-of-tree, GPL-2.0)
Pins user pages (e.g. page-cache folios behind a read-only `MAP_SHARED` mmap of a GGUF file) with `FOLL_LONGTERM`
and exports them as a dma-buf (`/dev/strixpc`, `STRIXPC_IOC_EXPORT`). Imported with `VK_EXT_external_memory_dma_buf`
this gives zero-copy weights as ordinary sg BOs: no amdgpu per-submit userptr revalidation, stock RADV/libdrm/amdgpu untouched.

    make                  # builds against /lib/modules/$(uname -r)/build
    sudo insmod strixpc.ko [max_pinned_mb=65536]
    sudo rmmod strixpc    # rollback; nothing is installed persistently, stock drivers unchanged

Pinned pages are unevictable until every importer releases the dma-buf; `max_pinned_mb` caps the total.
Not signed: loading taints the kernel (Secure Boot is not enforcing on this machine).
