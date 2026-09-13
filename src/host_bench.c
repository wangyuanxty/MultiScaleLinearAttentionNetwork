/* host_bench.c — how long does one multi-scale inference take on the host,
 * with process startup amortised away?
 *
 * The guest-side instruction count came out of the QEMU icount measurement as
 * ticks x 40. Before that number goes anywhere near the paper it needs an
 * independent sanity check, and the cheapest one is the same C source built
 * for x86: if the host does one inference in T seconds at a known rough
 * instruction rate, the guest count can be bracketed to within an order of
 * magnitude. A 10x disagreement would mean the icount reading is wrong.
 *
 * Build (from src/, so test_input_ms.csv resolves):
 *   gcc -O2 -I. -o host_bench host_bench.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gdn2_multiscale.c"
#include "gdn_ms_weights.h"
#include "ms_init.h"

static int read_csv(const char *path, float *out, int n) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    for (int i = 0; i < n; i++)
        if (fscanf(f, "%f", &out[i]) != 1) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    int n_rep = (argc > 1) ? atoi(argv[1]) : 200;

    GDN2MS_Variables m;
    float input[MS_L], output = 0.0f;
    double acc = 0.0;

    ms_init(&m);
    if (read_csv("test_input_ms.csv", input, MS_L) != 0) {
        fprintf(stderr, "cannot read test_input_ms.csv\n");
        return 1;
    }

    /* warm up, then time the loop rather than the process */
    for (int i = 0; i < 5; i++) GDN2MS_Infer(&m, input, &output);

    clock_t t0 = clock();
    for (int i = 0; i < n_rep; i++) {
        GDN2MS_Infer(&m, input, &output);
        acc += output;
    }
    clock_t t1 = clock();

    double per = (double)(t1 - t0) / CLOCKS_PER_SEC / n_rep;
    printf("reps=%d  per_infer=%.6f s  (%.3f ms)  checksum=%.6f\n",
           n_rep, per, per * 1e3, acc);
    return 0;
}
