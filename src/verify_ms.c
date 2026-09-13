/** verify_ms.c — PyTorch vs C comparison for the multi-scale model.
 *
 * Compile: cd src && gcc -O2 -o verify_ms verify_ms.c -lm
 * Run:     verify_ms      (reads test_input_ms.csv, prints the C output)
 *
 * The reference comes from the same checkpoint: export_gdn_weights.py built
 * the window at the checkpoint's SP and wrote PyTorch's answer to
 * test_py_out_ms.txt, so the two sides see byte-identical inputs.
 */
#include <stdio.h>
#include <stdlib.h>
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

int main(void) {
    GDN2MS_Variables m;
    float input[MS_L], output = 0.0f;

    ms_init(&m);
    if (read_csv("test_input_ms.csv", input, MS_L) != 0) {
        fprintf(stderr, "cannot read test_input_ms.csv\n");
        return 1;
    }
    if (GDN2MS_Infer(&m, input, &output) != GDN2_OK) {
        fprintf(stderr, "inference failed\n");
        return 1;
    }
    printf("C output: %.10f\n", output);
    printf("Compare with PyTorch: see test_py_out_ms.txt\n");
    return 0;
}
