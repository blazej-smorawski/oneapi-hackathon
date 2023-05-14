---
marp: true
style: uncover
_class: invert
---

# How to accelerate computing in 2023

Błażej Smorawski

---

# Problem - matrix multiplication
Presented approaches:
 * CPU only
 * GPU offloading
 * GPU scaleout

---

# SYCL

SYCL (pronounced ‘sickle’) is a royalty-free, cross-platform abstraction layer that:

* Enables code for heterogeneous and offload processors to be written using modern ISO C++ (at least C++ 17).
* Provides APIs and abstractions to find devices (e.g. CPUs, GPUs, FPGAs) on which code can be executed, and to manage data resources and code execution on those devices.

---

![](https://www.khronos.org/assets/uploads/apis/2022-sycl-diagram.jpg)

---

![](https://www.khronos.org/assets/uploads/apis/2020-05-sycl-landing-page-02a_1.jpg)