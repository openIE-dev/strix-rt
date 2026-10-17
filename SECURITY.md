# Security

strix-rt is research code meant to run on your own machine. It has no network service except the experimental OpenAI-style
servers (`strix-serve`, `strix-serve1`, `kev_serve`), which bind to 127.0.0.1 by default and are not hardened: please don't
expose them to untrusted networks. Model files are parsed with minimal validation, so only load GGUF files you trust.
The optional kernel module `kmod/strixpc` runs in the kernel; load it only on a machine you control.

If you find a security problem, please email **contact@openie.dev** instead of opening a public issue. We read every report and
will credit you if you like, but this is a small, read-only research project with no support promise or fixed response time.
