#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define TOTAL_SIZE (64 * 1024)  // 1 MB per iteration
#define CHUNK_SIZE (64 * 1024)  // 64 KB per splice call
#define CHUNKS (TOTAL_SIZE / CHUNK_SIZE)
#define ITERS 1000

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

ssize_t splice_all(int fd_in, off_t* off_in, int fd_out, off_t* off_out,
                   size_t len, unsigned int flags) {
  size_t total = 0;
  while (total < len) {
    ssize_t ret = splice(fd_in, off_in, fd_out, off_out, len - total, flags);
    if (ret < 0) {
      if (errno == EINTR) continue;
      perror("splice");
      return -1;
    } else if (ret == 0) {
      fprintf(stderr, "splice: unexpected EOF\n");
      return -1;
    }
    total += ret;
  }
  return total;
}

int main() {
  int pipefd[2];
  struct timespec start, end;

  pin_to_core(0);

  if (pipe(pipefd) < 0) {
    perror("pipe");
    exit(EXIT_FAILURE);
  }

  // Create a memfd with 1 MB of content
  int memfd = syscall(SYS_memfd_create, "splice_tmp", 0);
  if (memfd < 0) {
    perror("memfd_create");
    exit(EXIT_FAILURE);
  }

  // Write 1MB of content into memfd
  char* buf = (char*)malloc(CHUNK_SIZE);
  memset(buf, 0xBB, CHUNK_SIZE);
  for (int i = 0; i < CHUNKS; i++) {
    if (write(memfd, buf, CHUNK_SIZE) != CHUNK_SIZE) {
      perror("write to memfd");
      exit(EXIT_FAILURE);
    }
  }

  // Benchmark loop
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int iter = 0; iter < ITERS; iter++) {
    lseek(memfd, 0, SEEK_SET);  // reset memfd read pointer
    for (int i = 0; i < CHUNKS; i++) {
      if (splice_all(memfd, NULL, pipefd[1], NULL, CHUNK_SIZE, 0) < 0) exit(1);
      if (splice_all(pipefd[0], NULL, memfd, NULL, CHUNK_SIZE, 0) < 0) exit(1);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed = timespec_diff_sec(start, end);
  double total_MB =
      (double)(TOTAL_SIZE * ITERS * 2) / (1024 * 1024);  // write + read
  double throughput = total_MB / elapsed;

  printf("Spliced %.2f MB in %.3f seconds\n", total_MB, elapsed);
  printf("Throughput (zero-copy): %.2f MB/s\n", throughput);

  return 0;
}
