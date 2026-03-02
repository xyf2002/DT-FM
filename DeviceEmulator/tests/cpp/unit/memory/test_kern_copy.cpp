#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CHUNK_SIZE (32 * 1024)  // 32 KB per write/read
#define TOTAL_SIZE (32 * 1024)  // 1 MB total per iteration
#define ITERS 10000
#define CHUNKS_PER_ITER (TOTAL_SIZE / CHUNK_SIZE)

static inline double timespec_diff_sec(struct timespec start,
                                       struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void pin_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

ssize_t write_all(int fd, char* buf, size_t size) {
  size_t total = 0;
  while (total < size) {
    ssize_t ret = write(fd, buf + total, size - total);
    if (ret <= 0) return ret;
    total += ret;
  }
  return total;
}

ssize_t read_all(int fd, char* buf, size_t size) {
  size_t total = 0;
  while (total < size) {
    ssize_t ret = read(fd, buf + total, size - total);
    if (ret <= 0) return ret;
    total += ret;
  }
  return total;
}

int main() {
  int pipefd[2];
  char* write_buf = (char*)aligned_alloc(64, CHUNK_SIZE);
  char* read_buf = (char*)aligned_alloc(64, CHUNK_SIZE);
  struct timespec start, end;

  memset(write_buf, 0xAA, CHUNK_SIZE);

  pin_to_core(0);
  mlock(write_buf, CHUNK_SIZE);
  mlock(read_buf, CHUNK_SIZE);

  if (pipe(pipefd) == -1) {
    perror("pipe");
    exit(1);
  }

  // warm-up
  for (int i = 0; i < CHUNKS_PER_ITER; i++) {
    write_all(pipefd[1], write_buf, CHUNK_SIZE);
    read_all(pipefd[0], read_buf, CHUNK_SIZE);
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < ITERS; ++i) {
    for (int j = 0; j < CHUNKS_PER_ITER; j++) {
      write_all(pipefd[1], write_buf, CHUNK_SIZE);
    }
    for (int j = 0; j < CHUNKS_PER_ITER; j++) {
      read_all(pipefd[0], read_buf, CHUNK_SIZE);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed = timespec_diff_sec(start, end);
  double total_MB =
      (double)(TOTAL_SIZE * ITERS * 2) / (1024 * 1024);  // write + read
  double throughput = total_MB / elapsed;

  printf("Copied %.2f MB in %.3f seconds\n", total_MB, elapsed);
  printf("Throughput: %.2f MB/s\n", throughput);
  return 0;
}
