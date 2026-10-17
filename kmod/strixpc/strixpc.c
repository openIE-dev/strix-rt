// SPDX-License-Identifier: GPL-2.0
/*
 * strixpc - pin user pages (typically page-cache folios behind a read-only file mmap) with FOLL_LONGTERM
 * and export them as a dma-buf. Lets Vulkan (VK_EXT_external_memory_dma_buf) map GGUF weights zero-copy as a
 * normal sg BO, avoiding amdgpu's per-submit userptr revalidation (hmm_range_fault on every CS).
 * Pinned pages are unevictable until the dma-buf is released (the runtime's memory tier owns that budget).
 */
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/scatterlist.h>
#include <linux/atomic.h>
#include "strixpc.h"

MODULE_IMPORT_NS("DMA_BUF");

static atomic64_t pinned_pages = ATOMIC64_INIT(0);
static unsigned long max_pinned_mb = 65536;
module_param(max_pinned_mb, ulong, 0644);
MODULE_PARM_DESC(max_pinned_mb, "max MiB pinned across all exports");

struct spc_buf { struct page **pages; unsigned long npages; };

static struct sg_table *spc_map(struct dma_buf_attachment *at, enum dma_data_direction dir)
{
	struct spc_buf *b = at->dmabuf->priv;
	struct sg_table *sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	int r;

	if (!sgt)
		return ERR_PTR(-ENOMEM);
	r = sg_alloc_table_from_pages_segment(sgt, b->pages, b->npages, 0, b->npages << PAGE_SHIFT, UINT_MAX, GFP_KERNEL);
	if (r)
		goto free;
	r = dma_map_sgtable(at->dev, sgt, dir, DMA_ATTR_SKIP_CPU_SYNC);
	if (r)
		goto table;
	return sgt;
table:
	sg_free_table(sgt);
free:
	kfree(sgt);
	return ERR_PTR(r);
}

static void spc_unmap(struct dma_buf_attachment *at, struct sg_table *sgt, enum dma_data_direction dir)
{
	dma_unmap_sgtable(at->dev, sgt, dir, DMA_ATTR_SKIP_CPU_SYNC);
	sg_free_table(sgt);
	kfree(sgt);
}

static void spc_release(struct dma_buf *d)
{
	struct spc_buf *b = d->priv;

	unpin_user_pages(b->pages, b->npages);
	atomic64_sub(b->npages, &pinned_pages);
	kvfree(b->pages);
	kfree(b);
}

static int spc_mmap(struct dma_buf *d, struct vm_area_struct *vma)
{
	return -EINVAL; /* CPU access goes through the original mmap */
}

static const struct dma_buf_ops spc_ops = {
	.map_dma_buf = spc_map,
	.unmap_dma_buf = spc_unmap,
	.release = spc_release,
	.mmap = spc_mmap,
};

static long spc_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct strixpc_export e;
	DEFINE_DMA_BUF_EXPORT_INFO(exp);
	struct spc_buf *b;
	struct dma_buf *d;
	unsigned long got = 0;
	long r;

	if (cmd != STRIXPC_IOC_EXPORT)
		return -ENOTTY;
	if (copy_from_user(&e, (void __user *)arg, sizeof(e)))
		return -EFAULT;
	if (!e.len || ((e.addr | e.len) & ~PAGE_MASK) || e.flags)
		return -EINVAL;
	b = kzalloc(sizeof(*b), GFP_KERNEL);
	if (!b)
		return -ENOMEM;
	b->npages = e.len >> PAGE_SHIFT;
	if (atomic64_add_return(b->npages, &pinned_pages) > (long)(max_pinned_mb << (20 - PAGE_SHIFT))) {
		r = -ENOSPC;
		goto unacct;
	}
	b->pages = kvmalloc_array(b->npages, sizeof(struct page *), GFP_KERNEL);
	if (!b->pages) {
		r = -ENOMEM;
		goto unacct;
	}
	while (got < b->npages) {
		r = pin_user_pages_fast(e.addr + (got << PAGE_SHIFT), min(b->npages - got, 65536UL), FOLL_LONGTERM, b->pages + got);
		if (r <= 0) {
			if (!r)
				r = -EFAULT;
			goto unpin;
		}
		got += r;
	}
	exp.ops = &spc_ops;
	exp.size = e.len;
	exp.flags = O_RDWR;
	exp.priv = b;
	d = dma_buf_export(&exp);
	if (IS_ERR(d)) {
		r = PTR_ERR(d);
		goto unpin;
	}
	e.fd = dma_buf_fd(d, O_CLOEXEC);
	if (e.fd < 0) {
		r = e.fd;
		dma_buf_put(d); /* release() unpins */
		return r;
	}
	if (copy_to_user((void __user *)arg, &e, sizeof(e)))
		return -EFAULT; /* fd stays installed; caller leaks it, acceptable for a debug path */
	return 0;
unpin:
	unpin_user_pages(b->pages, got);
	kvfree(b->pages);
unacct:
	atomic64_sub(b->npages, &pinned_pages);
	kfree(b);
	return r;
}

static const struct file_operations spc_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = spc_ioctl,
	.compat_ioctl = spc_ioctl,
};

static struct miscdevice spc_dev = { .minor = MISC_DYNAMIC_MINOR, .name = "strixpc", .fops = &spc_fops, .mode = 0666 };

static int __init spc_init(void) { return misc_register(&spc_dev); }
static void __exit spc_exit(void) { misc_deregister(&spc_dev); }
module_init(spc_init);
module_exit(spc_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("strix-rt: export pinned page-cache pages as dma-buf");
