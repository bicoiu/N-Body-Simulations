#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

__global__
void computeForcesCUDA(Particle* p, int n, double dt) {
    const double G = 6.67430e-11;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        double fx = 0.0, fy = 0.0, fz = 0.0;

        double xi = p[i].x;
        double yi = p[i].y;
        double zi = p[i].z;
        double mi = p[i].mass;

        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dx = p[j].x - xi;
            double dy = p[j].y - yi;
            double dz = p[j].z - zi;

            double distSqr = dx * dx + dy * dy + dz * dz + 1e-9;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            double F = G * mi * p[j].mass * invDist3;

            fx += F * dx;
            fy += F * dy;
            fz += F * dz;
        }

        p[i].vx += dt * fx / mi;
        p[i].vy += dt * fy / mi;
        p[i].vz += dt * fz / mi;
    }
}

__global__
void updatePositionsCUDA(Particle* p, int n, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        p[i].x += dt * p[i].vx;
        p[i].y += dt * p[i].vy;
        p[i].z += dt * p[i].vz;
    }
}

// checksum kernel: reduction in shared + atomicAdd on float (supported on 920MX)
__global__
void checksumKernel(const Particle* p, int n, float* out) {
    extern __shared__ float buf[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += (float)(p[i].x + p[i].y + p[i].z);
    }

    buf[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, buf[0]);
}

static void initializeParticles(std::vector<Particle>& h) {
    for (auto& x : h) {
        x.x = drand48() * 100.0;
        x.y = drand48() * 100.0;
        x.z = drand48() * 100.0;
        x.vx = drand48();
        x.vy = drand48();
        x.vz = drand48();
        x.mass = drand48() * 10.0 + 1.0;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: ./nbody_cuda N STEPS THREADS BLOCKS\n";
        std::cout << "Example: ./nbody_cuda 3000 3000 256 80\n";
        return 1;
    }

    int N = atoi(argv[1]);
    long long STEPS = atoll(argv[2]);
    int threads = atoi(argv[3]);
    int blocks = atoi(argv[4]);

    if (N <= 0 || STEPS <= 0) {
        std::cerr << "N and STEPS must be > 0\n";
        return 1;
    }
    if (threads <= 0 || threads > 1024) {
        std::cerr << "THREADS must be in range [1..1024]\n";
        return 1;
    }
    if (blocks <= 0 || blocks > 65535) {
        std::cerr << "BLOCKS must be in range [1..65535]\n";
        return 1;
    }

    // show GPU info
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "No CUDA device found\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name
              << " | SMs=" << prop.multiProcessorCount
              << " | GlobalMem=" << (prop.totalGlobalMem / (1024 * 1024)) << " MiB\n";

    double dt = 0.01;

    // host init
    std::vector<Particle> h_particles(N);
    initializeParticles(h_particles);

    // device alloc + copy
    Particle* d_particles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles.data(),
                          N * sizeof(Particle), cudaMemcpyHostToDevice));

    // ---------------- WARM-UP ----------------
    for (int i = 0; i < 5; i++) {
        computeForcesCUDA<<<blocks, threads>>>(d_particles, N, dt);
        CUDA_CHECK(cudaGetLastError());

        updatePositionsCUDA<<<blocks, threads>>>(d_particles, N, dt);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- TIMING (KERNELS ONLY) ----------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (long long step = 0; step < STEPS; step++) {
        computeForcesCUDA<<<blocks, threads>>>(d_particles, N, dt);
        CUDA_CHECK(cudaGetLastError());

        updatePositionsCUDA<<<blocks, threads>>>(d_particles, N, dt);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // ---------------- CHECKSUM ----------------
    float* d_sum = nullptr;
    float h_sum = 0.0f;

    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    size_t shmem = (size_t)threads * sizeof(float);
    checksumKernel<<<blocks, threads, shmem>>>(d_particles, N, d_sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N=" << N
              << " STEPS=" << STEPS
              << " THREADS=" << threads
              << " BLOCKS=" << blocks << "\n";

    std::cout << "Timp kernel GPU: " << ms << " ms ("
              << (ms / 1000.0) << " sec)\n";

    std::cout << std::setprecision(8);
    std::cout << "Checksum: " << h_sum << "\n";

    // cleanup
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
