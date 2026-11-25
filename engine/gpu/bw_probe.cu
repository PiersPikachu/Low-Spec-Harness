// bw_probe.cu - measures what the planner needs from the GPU side: free/total VRAM and pinned host<->device bandwidth.
// prints one JSON line.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"error\":\"%s\"}\n", cudaGetErrorString(e_)); return 1; } } while (0)
static double run(void *dst, const void *src, size_t n, cudaMemcpyKind k, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaMemcpyAsync(dst, src, n, k); cudaDeviceSynchronize();            // warm-up
    cudaEventRecord(a); for (int i = 0; i < reps; i++) cudaMemcpyAsync(dst, src, n, k); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return (double)n * reps / 1e9 / (ms / 1e3);
}
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    size_t n = 256u << 20; void *h, *d, *hp = malloc(n); CK(cudaHostAlloc(&h, n, cudaHostAllocDefault)); CK(cudaMalloc(&d, n));
    for (size_t i = 0; i < n; i += 4096) ((char *)h)[i] = 1; for (size_t i = 0; i < n; i += 4096) ((char *)hp)[i] = 1;
    double h2d = run(d, h, n, cudaMemcpyHostToDevice, 10), d2h = run(h, d, n, cudaMemcpyDeviceToHost, 10), pg = run(d, hp, n, cudaMemcpyHostToDevice, 4);
    double d2d = run((char *)d + 0, d, n / 2, cudaMemcpyDeviceToDevice, 20) * 2.0;   // read+write traffic
    printf("{\"gpu\":\"%s\",\"sm\":\"%d%d\",\"vram_total_mb\":%.0f,\"vram_free_mb\":%.0f,\"h2d_pinned_GBps\":%.2f,\"d2h_pinned_GBps\":%.2f,\"h2d_pageable_GBps\":%.2f,\"d2d_GBps\":%.0f}\n",
           p.name, p.major, p.minor, tot / 1048576.0, fr / 1048576.0, h2d, d2h, pg, d2d);
    return 0;
}
