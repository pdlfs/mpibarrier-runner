/*
 * Copyright (c) 2018, Carnegie Mellon University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * mpibarrier-runner.cc
 */
#include <errno.h>
#include <getopt.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <signal.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <mpi.h>

/*
 * helper/utility functions, included inline here so we are self-contained
 * in one single source file...
 */
static char* argv0; /* argv[0], program name */
static int myrank = 0;

/*
 * vcomplain/complain about something.  if ret is non-zero we exit(ret)
 * after complaining.  if r0only is set, we only print if myrank == 0.
 */
static void vcomplain(int ret, int r0only, const char* format, va_list ap) {
  if (!r0only || myrank == 0) {
    fprintf(stderr, "%s: ", argv0);
    vfprintf(stderr, format, ap);
    fprintf(stderr, "\n");
  }
  if (ret) {
    MPI_Finalize();
    exit(ret);
  }
}

static void complain(int ret, int r0only, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  vcomplain(ret, r0only, format, ap);
  va_end(ap);
}

/*
 * abort with a fatal message
 */
#define FATAL(msg) fatal(__FILE__, __LINE__, msg)
static void fatal(const char* f, int d, const char* msg) {
  fprintf(stderr, "=== ABORT === ");
  fprintf(stderr, "%s (%s:%d)", msg, f, d);
  fprintf(stderr, "\n");
  abort();
}

/*
 * default values
 */
#define DEF_TIMEOUT 120     /* alarm timeout */
#define DEF_BARR_SLEEP 5000 /* 5000 microseconds */
#define DEF_WAIT 10

/*
 * gs: shared global data (e.g. from the command line)
 */
static struct gs {
  int size;    /* world size (from MPI) */
  int timeout; /* alarm timeout */
  int barrier_sleep;
  int wait;
} g;

/*
 * alarm signal handler
 */
static void sigalarm(int foo) {
  fprintf(stderr, "SIGALRM detected (%d)\n", myrank);
  fprintf(stderr, "Alarm clock\n");
  MPI_Finalize();
  exit(1);
}

/*
 * we want a single number
 */
static inline double fl(const struct timeval* tv) {
  return (1.0 * tv->tv_sec + 1.0 * tv->tv_usec / 1000 / 1000);
}

/*
 * usage
 */
static void usage(const char* msg) {
  /* only have rank 0 print usage error message */
  if (myrank) goto skip_prints;

  if (msg) fprintf(stderr, "%s: %s\n", argv0, msg);
  fprintf(stderr, "usage: %s [options]\n", argv0);
  fprintf(stderr, "\noptions:\n");
  fprintf(stderr, "\t-t sec      timeout (alarm), in seconds\n");

skip_prints:
  MPI_Finalize();
  exit(1);
}

/*
 * forward prototype decls.
 */
static void MPI_Barrier_x(MPI_Comm comm);
static void doit();

/*
 * main program.
 */
int main(int argc, char* argv[]) {
  int ch;

  argv0 = argv[0];

  /* mpich says we should call this early as possible */
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    FATAL("!MPI_Init");
  }

  /* we want lines!! */
  setlinebuf(stdout);

  /* setup default to zero/null, except as noted below */
  memset(&g, 0, sizeof(g));
  if (MPI_Comm_rank(MPI_COMM_WORLD, &myrank) != MPI_SUCCESS)
    FATAL("!MPI_Comm_rank");
  if (MPI_Comm_size(MPI_COMM_WORLD, &g.size) != MPI_SUCCESS)
    FATAL("!MPI_Comm_size");

  g.timeout = DEF_TIMEOUT;
  g.barrier_sleep = DEF_BARR_SLEEP;
  g.wait = DEF_WAIT;

  while ((ch = getopt(argc, argv, "t:")) != -1) {
    switch (ch) {
      case 't':
        g.timeout = atoi(optarg);
        if (g.timeout < 0) usage("bad timeout");
        break;
      default:
        usage(NULL);
    }
  }

  if (myrank == 0) {
    printf("== Program options:\n");
    printf("MPI_rank   = %d\n", myrank);
    printf("MPI_size   = %d\n", g.size);
    printf("barrier sleep  = %d microsecs\n", g.barrier_sleep);
    printf("0th_rank wait  = %d secs\n", g.wait);
    printf("timeout    = %d secs\n", g.timeout);
    printf("\n");
  }

  signal(SIGALRM, sigalarm);
  alarm(g.timeout);

  doit();

  MPI_Finalize();

  return 0;
}

static void MPI_Barrier_x(MPI_Comm comm) {
  MPI_Request req;
  MPI_Status status;
  int ok = 0;

  if (MPI_Ibarrier(comm, &req) != MPI_SUCCESS) complain(1, 0, "!MPI_Ibarrier");
  while (!ok) {
    usleep(g.barrier_sleep);
    if (MPI_Test(&req, &ok, &status) != MPI_SUCCESS)
      complain(1, 0, "!MPI_Test");
  }
}

static void doit() {
  struct rusage befo;
  struct rusage afte;
  int i;

  if (myrank == 0) {
    sleep(g.wait); /* the 0th-rank performs a long wait */
  }

  if (getrusage(RUSAGE_SELF, &befo) != 0) {
    FATAL("!getrusage");
  }
  MPI_Barrier_x(MPI_COMM_WORLD);
  if (getrusage(RUSAGE_SELF, &afte) != 0) {
    FATAL("!getrusage");
  }

  for (i = 0; i < g.size; i++) {
    if (myrank == i) {
      MPI_Barrier(MPI_COMM_WORLD);
      printf("\nRank %d:\n", myrank);
      printf("usr/sys=%.3f/%.3f s\n", fl(&afte.ru_utime) - fl(&befo.ru_utime),
             fl(&afte.ru_stime) - fl(&befo.ru_stime));
    }
  }
}
