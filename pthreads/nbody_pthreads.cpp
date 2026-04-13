#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <pthread.h>

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

//globale
static Particle* g_particles = nullptr;
static int g_N = 0;
static double g_dt = 0.01;

struct ThreadArgs {
    int start;
    int end;
};

void initializeParticles(std::vector<Particle>& p, int n) {
    for (int i = 0; i < n; i++) {
        p[i].x = drand48() * 100.0;
        p[i].y = drand48() * 100.0;
        p[i].z = drand48() * 100.0;

        p[i].vx = drand48();
        p[i].vy = drand48();
        p[i].vz = drand48();

        p[i].mass = drand48() * 10.0 + 1.0;
    }
}

void* computeForcesThread(void* arg) {
    ThreadArgs* targs = static_cast<ThreadArgs*>(arg);
    int start = targs->start;
    int end   = targs->end;

    const double G = 6.67430e-11;

    for (int i = start; i < end; i++) {
        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (int j = 0; j < g_N; j++) {
            if (i == j) continue;

            double dx = g_particles[j].x - g_particles[i].x;
            double dy = g_particles[j].y - g_particles[i].y;
            double dz = g_particles[j].z - g_particles[i].z;

            double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
            double invDist = 1.0 / std::sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            double F = G * g_particles[i].mass * g_particles[j].mass * invDist3;

            fx += F * dx;
            fy += F * dy;
            fz += F * dz;
        }

        g_particles[i].vx += g_dt * fx / g_particles[i].mass;
        g_particles[i].vy += g_dt * fy / g_particles[i].mass;
        g_particles[i].vz += g_dt * fz / g_particles[i].mass;
    }

    return nullptr;
}

void* updatePositionsThread(void* arg) {
    ThreadArgs* targs = static_cast<ThreadArgs*>(arg);
    int start = targs->start;
    int end   = targs->end;

    for (int i = start; i < end; i++) {
        g_particles[i].x += g_dt * g_particles[i].vx;
        g_particles[i].y += g_dt * g_particles[i].vy;
        g_particles[i].z += g_dt * g_particles[i].vz;
    }

    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./nbody_pth_step2 N STEPS [THREADS]\n";
        return 1;
    }

    int N       = std::atoi(argv[1]);
    int STEPS   = std::atoi(argv[2]);
    int THREADS = (argc >= 4) ? std::atoi(argv[3]) : 4;
    if (THREADS <= 0) THREADS = 1;

    g_N  = N;
    g_dt = 0.01;

    std::vector<Particle> particles(N);
    initializeParticles(particles, N);
    g_particles = particles.data();

    std::cout << "Numar particule: " << N
              << ", pasi: " << STEPS
              << ", threads: " << THREADS << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; step++) {
        //compute
        {
            std::vector<pthread_t> threads(THREADS);
            std::vector<ThreadArgs> targs(THREADS);

            int base = N / THREADS;
            int rest = N % THREADS;
            int current = 0;

            for (int t = 0; t < THREADS; t++) {
                int chunk = base + (t < rest ? 1 : 0);
                targs[t].start = current;
                targs[t].end   = current + chunk;
                current += chunk;

                pthread_create(&threads[t], nullptr, computeForcesThread, &targs[t]);
            }

            for (int t = 0; t < THREADS; t++) {
                pthread_join(threads[t], nullptr);
            }
        }

        // calcul pos
        {
            std::vector<pthread_t> threads(THREADS);
            std::vector<ThreadArgs> targs(THREADS);

            int base = N / THREADS;
            int rest = N % THREADS;
            int current = 0;

            for (int t = 0; t < THREADS; t++) {
                int chunk = base + (t < rest ? 1 : 0);
                targs[t].start = current;
                targs[t].end   = current + chunk;
                current += chunk;

                pthread_create(&threads[t], nullptr, updatePositionsThread, &targs[t]);
            }

            for (int t = 0; t < THREADS; t++) {
                pthread_join(threads[t], nullptr);
            }
        }

        if (step % 100 == 0)
            std::cout << "Step " << step << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Timp total simulare: "
              << elapsed.count() << " secunde.\n";

    return 0;
}