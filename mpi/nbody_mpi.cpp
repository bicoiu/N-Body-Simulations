#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double mass;
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

void computeRange(int rank, int size, int n, int& start, int& end) {
    int chunk = n / size;
    int r = n % size;

    start = rank * chunk + std::min(rank, r);
    end   = start + chunk + (rank < r ? 1 : 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (!rank) std::cerr << "Usage: mpirun ./nbody_mpi N STEPS\n";
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int STEPS = atoi(argv[2]);
    double dt = 0.01;

    std::vector<Particle> particles(N);

    if (rank == 0)
        initializeParticles(particles, N);

    MPI_Bcast(particles.data(), N * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);

    int start, end;
    computeRange(rank, size, N, start, end);

    const double G = 6.67430e-11;

    double t0 = MPI_Wtime();

    for (int step = 0; step < STEPS; step++) {
        for (int i = start; i < end; i++) {
            double fx = 0, fy = 0, fz = 0;

            for (int j = 0; j < N; j++) {
                if (i == j) continue;

                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;

                double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                double F = G * particles[i].mass * particles[j].mass * invDist3;

                fx += F * dx;
                fy += F * dy;
                fz += F * dz;
            }

            particles[i].vx += dt * fx / particles[i].mass;
            particles[i].vy += dt * fy / particles[i].mass;
            particles[i].vz += dt * fz / particles[i].mass;

            particles[i].x += dt * particles[i].vx;
            particles[i].y += dt * particles[i].vy;
            particles[i].z += dt * particles[i].vz;
        }

        MPI_Allgather(
            MPI_IN_PLACE,
            0,
            MPI_DATATYPE_NULL,
            particles.data(),
            (end - start) * sizeof(Particle),
            MPI_BYTE,
            MPI_COMM_WORLD
        );
    }

    double t1 = MPI_Wtime();

    if (rank == 0)
        std::cout << "MPI time: " << t1 - t0 << " sec\n";

    MPI_Finalize();
    return 0;
}