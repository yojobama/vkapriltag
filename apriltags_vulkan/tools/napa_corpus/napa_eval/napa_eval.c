/* Native runner for the NAPA detector over the test corpus: the C counterpart of napa_run.py.
 * usage: napa_eval ROOT OUT_DIR [--min-tag-px N] [--reps N] [--gap-ms MS] [--only SCENARIO] [--list-name NAME] */
#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define MKDIR(p) _mkdir(p)
#else
#include <time.h>
#define MKDIR(p) mkdir(p, 0777)
#endif

#include "napa_core.h"

#define MAX_OUT 64
#define MAX_PATH_LEN 1024

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e3 * (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec;
#endif
}

static void sleep_ms(double ms) {
#ifdef _WIN32
    Sleep((DWORD)(ms + 0.5));
#else
    struct timespec t;
    t.tv_sec = (time_t)(ms / 1e3);
    t.tv_nsec = (long)((ms - 1e3 * (double)t.tv_sec) * 1e6);
    nanosleep(&t, NULL);
#endif
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int file_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static const char *base_name(const char *p) {
    const char *s = p;
    for (; *p; p++)
        if (*p == '/' || *p == '\\') s = p + 1;
    return s;
}

static unsigned char *read_pgm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int maxval = 0;
    unsigned char *px = NULL;
    if (fscanf(f, "P5 %d %d %d", w, h, &maxval) == 3 && maxval == 255 && *w > 0 && *h > 0 && fgetc(f) != EOF) {
        size_t n = (size_t)*w * (size_t)*h;
        px = (unsigned char *)malloc(n);
        if (px && fread(px, 1, n, f) != n) {
            free(px);
            px = NULL;
        }
    }
    fclose(f);
    return px;
}

/* Centre of the homography mapping (+-1, +-1) to the corners, as NapaDetector.center does. */
static int tag_centre(const double *c, double *cx, double *cy) {
    static const double u[4] = {1, -1, -1, 1}, v[4] = {1, 1, -1, -1};
    double a[8][9];
    for (int i = 0; i < 4; i++) {
        double x = c[2 * i], y = c[2 * i + 1];
        double r0[9] = {u[i], v[i], 1, 0, 0, 0, -u[i] * x, -v[i] * x, x};
        double r1[9] = {0, 0, 0, u[i], v[i], 1, -u[i] * y, -v[i] * y, y};
        memcpy(a[2 * i], r0, sizeof r0);
        memcpy(a[2 * i + 1], r1, sizeof r1);
    }
    for (int k = 0; k < 8; k++) {
        int p = k;
        for (int i = k + 1; i < 8; i++)
            if (fabs(a[i][k]) > fabs(a[p][k])) p = i;
        if (a[p][k] == 0) return -1;
        if (p != k) {
            double t[9];
            memcpy(t, a[p], sizeof t);
            memcpy(a[p], a[k], sizeof t);
            memcpy(a[k], t, sizeof t);
        }
        for (int i = k + 1; i < 8; i++) {
            double m = a[i][k] / a[k][k];
            for (int j = k; j < 9; j++) a[i][j] -= m * a[k][j];
        }
    }
    double s[8];
    for (int i = 7; i >= 0; i--) {
        double t = a[i][8];
        for (int j = i + 1; j < 8; j++) t -= a[i][j] * s[j];
        s[i] = t / a[i][i];
    }
    *cx = s[2];
    *cy = s[5];
    return 0;
}

static char **read_lines(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    char line[MAX_PATH_LEN];
    char **v = NULL;
    *n = 0;
    if (!f) return NULL;
    while (fgets(line, sizeof line, f)) {
        size_t l = strlen(line);
        while (l && (line[l - 1] == '\n' || line[l - 1] == '\r' || line[l - 1] == ' ' || line[l - 1] == '\t')) line[--l] = 0;
        const char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) continue;
        v = (char **)realloc(v, (size_t)(*n + 1) * sizeof *v);
        v[(*n)++] = strdup(s);
    }
    fclose(f);
    return v;
}

static char **list_dir(const char *dir, int want_dirs, int *n) {
    DIR *d = opendir(dir);
    struct dirent *e;
    char **v = NULL;
    *n = 0;
    if (!d) return NULL;
    while ((e = readdir(d))) {
        char p[MAX_PATH_LEN];
        size_t l = strlen(e->d_name);
        struct stat st;
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (stat(p, &st) != 0) continue;
        if (want_dirs ? !S_ISDIR(st.st_mode) : (l < 4 || strcmp(e->d_name + l - 4, ".pgm") != 0)) continue;
        v = (char **)realloc(v, (size_t)(*n + 1) * sizeof *v);
        v[(*n)++] = want_dirs ? strdup(e->d_name) : strdup(p);
    }
    closedir(d);
    if (*n) qsort(v, (size_t)*n, sizeof *v, cmp_str);
    return v;
}

int main(int argc, char **argv) {
    const char *root = NULL, *outdir = NULL, *only = NULL, *list_name = "list.txt";
    int min_tag_px = 32, reps = 5;
    double gap_ms = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--min-tag-px") && i + 1 < argc) min_tag_px = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gap-ms") && i + 1 < argc) gap_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--list-name") && i + 1 < argc) list_name = argv[++i];
        else if (!root) root = argv[i];
        else if (!outdir) outdir = argv[i];
        else root = NULL;
    }
    if (!root || !outdir || reps < 1) {
        fprintf(stderr, "usage: %s ROOT OUT_DIR [--min-tag-px N] [--reps N] [--gap-ms MS] [--only SCENARIO] [--list-name NAME]\n", argv[0]);
        return 2;
    }
    MKDIR(outdir);
    char tpath[MAX_PATH_LEN];
    snprintf(tpath, sizeof tpath, "%s/timings.json", outdir);
    FILE *tj = fopen(tpath, "w");
    if (!tj) {
        fprintf(stderr, "cannot write %s\n", tpath);
        return 1;
    }
    fputc('{', tj);
    int nsc, first_sc = 1;
    char **scen = list_dir(root, 1, &nsc);
    for (int si = 0; si < nsc; si++) {
        const char *s = scen[si];
        char dir[MAX_PATH_LEN], lst[MAX_PATH_LEN];
        int nl = 0;
        char **lines;
        if (only && strcmp(only, s)) continue;
        snprintf(dir, sizeof dir, "%s/%s", root, s);
        snprintf(lst, sizeof lst, "%s/%s", dir, list_name);
        lines = file_exists(lst) ? read_lines(lst, &nl) : list_dir(dir, 0, &nl);
        if (!nl) continue;
        const char *family = strcmp(s, "family16h5") ? "tag36h11" : "tag16h5";
        napa_family_layout lay;
        napa_detector *det = napa_create_family(family);
        if (!det || napa_family_info(family, &lay) != 0) {
            fprintf(stderr, "cannot create detector for %s\n", family);
            return 1;
        }
        napa_params prm = {.min_tag_px = min_tag_px, .max_hamming = lay.recommended_hamming, .min_contrast = 20,
                           .min_border_cells = 22, .sharpening_x100 = 25};
        char opath[MAX_PATH_LEN];
        snprintf(opath, sizeof opath, "%s/%s.txt", outdir, s);
        FILE *of = fopen(opath, "w");
        if (!of) {
            fprintf(stderr, "cannot write %s\n", opath);
            return 1;
        }
        if (!first_sc) fputc(',', tj);
        first_sc = 0;
        fprintf(tj, "\"%s\": [", s);
        int ntim = 0;
        double *ts = (double *)malloc((size_t)reps * sizeof *ts);
        for (int i = 0; i < nl; i++) {
            char alt[MAX_PATH_LEN];
            const char *p = lines[i];
            int w, h, cnt = 0;
            napa_detection res[MAX_OUT];
            if (!file_exists(p)) {
                snprintf(alt, sizeof alt, "%s/%s", dir, base_name(p));
                p = alt;
            }
            unsigned char *img = read_pgm(p, &w, &h);
            if (!img) continue;
            for (int r = 0; r < reps; r++) {
                if (gap_ms > 0) sleep_ms(gap_ms);
                double t0 = now_ms();
                cnt = (w < 16 || h < 16) ? 0 : napa_detect(det, img, w, h, w, &prm, res, MAX_OUT);
                ts[r] = now_ms() - t0;
                if (cnt < 0) {
                    fprintf(stderr, "napa_detect failed (%d) on %s\n", cnt, p);
                    return 1;
                }
            }
            double mn = ts[0];
            for (int r = 1; r < reps; r++)
                if (ts[r] < mn) mn = ts[r];
            qsort(ts, (size_t)reps, sizeof *ts, cmp_double);
            double med = reps % 2 ? ts[reps / 2] : 0.5 * (ts[reps / 2 - 1] + ts[reps / 2]);
            fprintf(tj, "%s[%d, %d, %.6f, %.6f]", ntim++ ? ", " : "", w, h, med, mn);
            fprintf(of, "F %d %d\n", i, cnt);
            for (int k = 0; k < cnt; k++) {
                static const int ord[4] = {1, 0, 3, 2};
                double cx = 0, cy = 0;
                tag_centre(res[k].corners, &cx, &cy);
                fprintf(of, "%d %d %.3f %.6f %.6f", res[k].id, res[k].hamming, (double)res[k].decision_margin, cx + 0.5, cy + 0.5);
                for (int j = 0; j < 4; j++)
                    fprintf(of, " %.6f %.6f", res[k].corners[2 * ord[j]] + 0.5, res[k].corners[2 * ord[j] + 1] + 0.5);
                fputc('\n', of);
            }
            free(img);
        }
        fprintf(tj, "]");
        fclose(of);
        free(ts);
        napa_destroy(det);
        printf("%s: %d frames\n", s, ntim);
    }
    fputc('}', tj);
    fclose(tj);
    return 0;
}
