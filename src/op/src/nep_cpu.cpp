#include "../include/nep_cpu.h"
#include <cmath>

void cpu_dev_apply_mic(const double* box, double& x12, double& y12, double& z12) {
    double sx12 = box[9] * x12 + box[10] * y12 + box[11] * z12;
    double sy12 = box[12] * x12 + box[13] * y12 + box[14] * z12;
    double sz12 = box[15] * x12 + box[16] * y12 + box[17] * z12;
    sx12 -= nearbyint(sx12);
    sy12 -= nearbyint(sy12);
    sz12 -= nearbyint(sz12);
    x12 = box[0] * sx12 + box[1] * sy12 + box[2] * sz12;
    y12 = box[3] * sx12 + box[4] * sy12 + box[5] * sz12;
    z12 = box[6] * sx12 + box[7] * sy12 + box[8] * sz12;
}

void find_fc(double rc, double rcinv, double d12, double& fc) {
    if (d12 < rc) {
        double x = d12 * rcinv;
        fc = 0.5 * cos(PI * x) + 0.5;
    } else {
        fc = 0.0;
    }
}

void find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5 * fc12;
  }
}

void find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn) {
    if (n == 0) {
        fn = fc12;
    } else if (n == 1) {
        double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
        fn = (x + 1.0) * 0.5 * fc12;
    } else {
        double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
        double t0 = 1.0;
        double t1 = x;
        double t2;
        for (int m = 2; m <= n; ++m) {
            t2 = 2.0 * x * t1 - t0;
            t0 = t1;
            t1 = t2;
        }
        fn = (t2 + 1.0) * 0.5 * fc12;
    }
}

void accumulate_s(const double d12, double x12, double y12, double z12, const double fn, double* s) {
    double d12inv = 1.0 / d12;
    x12 *= d12inv;
    y12 *= d12inv;
    z12 *= d12inv;
    double x12sq = x12 * x12;
    double y12sq = y12 * y12;
    double z12sq = z12 * z12;
    double x12sq_minus_y12sq = x12sq - y12sq;
    s[0] += z12 * fn;                                                            // Y10
    s[1] += x12 * fn;                                                            // Y11_real
    s[2] += y12 * fn;                                                            // Y11_imag
    s[3] += (3.0 * z12sq - 1.0) * fn;                                            // Y20
    s[4] += x12 * z12 * fn;                                                      // Y21_real
    s[5] += y12 * z12 * fn;                                                      // Y21_imag
    s[6] += x12sq_minus_y12sq * fn;                                              // Y22_real
    s[7] += 2.0 * x12 * y12 * fn;                                                // Y22_imag
    s[8] += (5.0 * z12sq - 3.0) * z12 * fn;                                      // Y30
    s[9] += (5.0 * z12sq - 1.0) * x12 * fn;                                      // Y31_real
    s[10] += (5.0 * z12sq - 1.0) * y12 * fn;                                     // Y31_imag
    s[11] += x12sq_minus_y12sq * z12 * fn;                                       // Y32_real
    s[12] += 2.0 * x12 * y12 * z12 * fn;                                         // Y32_imag
    s[13] += (x12 * x12 - 3.0 * y12 * y12) * x12 * fn;                           // Y33_real
    s[14] += (3.0 * x12 * x12 - y12 * y12) * y12 * fn;                           // Y33_imag
    s[15] += ((35.0 * z12sq - 30.0) * z12sq + 3.0) * fn;                         // Y40
    s[16] += (7.0 * z12sq - 3.0) * x12 * z12 * fn;                               // Y41_real
    s[17] += (7.0 * z12sq - 3.0) * y12 * z12 * fn;                               // Y41_imag
    s[18] += (7.0 * z12sq - 1.0) * x12sq_minus_y12sq * fn;                       // Y42_real
    s[19] += (7.0 * z12sq - 1.0) * x12 * y12 * 2.0 * fn;                         // Y42_imag
    s[20] += (x12sq - 3.0 * y12sq) * x12 * z12 * fn;                             // Y43_real
    s[21] += (3.0 * x12sq - y12sq) * y12 * z12 * fn;                             // Y43_imag
    s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq) * fn; // Y44_real
    s[23] += (4.0 * x12 * y12 * x12sq_minus_y12sq) * fn;                         // Y44_imag
}

void find_q(const int n_max_angular_plus_1, const int n, const double* s, double* q) {
    q[n] = C3B[0] * s[0] * s[0] + 2.0 * (C3B[1] * s[1] * s[1] + C3B[2] * s[2] * s[2]);
    q[n_max_angular_plus_1 + n] =
        C3B[3] * s[3] * s[3] + 2.0 * (C3B[4] * s[4] * s[4] + C3B[5] * s[5] * s[5] +
                                      C3B[6] * s[6] * s[6] + C3B[7] * s[7] * s[7]);
    q[2 * n_max_angular_plus_1 + n] =
        C3B[8] * s[8] * s[8] +
        2.0 * (C3B[9] * s[9] * s[9] + C3B[10] * s[10] * s[10] + C3B[11] * s[11] * s[11] +
               C3B[12] * s[12] * s[12] + C3B[13] * s[13] * s[13] + C3B[14] * s[14] * s[14]);
    q[3 * n_max_angular_plus_1 + n] =
        C3B[15] * s[15] * s[15] +
        2.0 * (C3B[16] * s[16] * s[16] + C3B[17] * s[17] * s[17] + C3B[18] * s[18] * s[18] +
               C3B[19] * s[19] * s[19] + C3B[20] * s[20] * s[20] + C3B[21] * s[21] * s[21] +
               C3B[22] * s[22] * s[22] + C3B[23] * s[23] * s[23]);
}

void find_q_with_4body(const int n_max_angular_plus_1, const int n, const double* s, double* q) {
    find_q(n_max_angular_plus_1, n, s, q);
    q[4 * n_max_angular_plus_1 + n] =
        C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
        C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
        C4B[4] * s[4] * s[5] * s[7];
}

void find_q_with_5body(const int n_max_angular_plus_1, const int n, const double* s, double* q) {
    find_q_with_4body(n_max_angular_plus_1, n, s, q);
    double s0_sq = s[0] * s[0];
    double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[5 * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                      C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
}
