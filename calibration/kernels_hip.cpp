// 海光 K100（HIP/ROCm）微基准：为仿真器校准采集原始测量值。
//
// 编译（在 ROCm 环境）：
//   hipcc -O2 -o kernels_hip kernels_hip.cpp -lrocblas
//
// 每个模式向 stdout 打印一行 JSON（错误走 stderr），供
// run_calibration.py 聚合。模式：
//   launch     空 kernel 启动开销
//   gemm       rocblas GEMM 有效算力（逐 K 扫描 -> 形状曲线）
//   copy       同设备大数组拷贝 -> 有效显存带宽
//   pingpong   双卡 ping-pong -> 逐对链路带宽 / 通信延迟
//   concurrency 两对不相交链路并发 vs 串行 -> 模型结构判定
//   overlap    同卡并发 GEMM + 拷贝 -> 计算-通信干扰系数
//   fusion     融合 vs 未融合 -> 每额外算子的融合开销
//
// 摩尔卡移植：把 hip* -> musa*、rocblas -> MUDNN/自研 GEMM，
// 编译命令换成 musa 的编译器（mcc/musacc），协议与聚合脚本不变。

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using std::map;
using std::string;

static void die(const string& msg) {
  fprintf(stderr, "[calib] error: %s\n", msg.c_str());
  exit(1);
}

static void chk(hipError_t e) {
  if (e != hipSuccess) {
    fprintf(stderr, "[calib] HIP error %d: %s\n", (int)e, hipGetErrorString(e));
    exit(1);
  }
}

static void chk_r(rocblas_status s) {
  if (s != rocblas_status_success) {
    fprintf(stderr, "[calib] rocblas error %d\n", (int)s);
    exit(1);
  }
}

static map<string, string> parse_args(int argc, char** argv) {
  map<string, string> m;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    if (a.rfind("--", 0) == 0) {
      string body = a.substr(2);
      size_t eq = body.find('=');
      if (eq != string::npos) {
        m[body.substr(0, eq)] = body.substr(eq + 1);
      } else if (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) {
        m[body] = argv[++i];  // --mode launch 空格分隔写法
      } else {
        m[body] = "1";
      }
    }
  }
  return m;
}

static int geti(const map<string, string>& m, const char* k, int dflt) {
  auto it = m.find(k);
  return it == m.end() ? dflt : atoi(it->second.c_str());
}

static double getf(const map<string, string>& m, const char* k, double dflt) {
  auto it = m.find(k);
  return it == m.end() ? dflt : atof(it->second.c_str());
}

static double event_ms(hipEvent_t s, hipEvent_t e) {
  float ms = 0.0f;
  chk(hipEventElapsedTime(&ms, s, e));
  return (double)ms;
}

__global__ void empty_kernel() {}

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void scale_kernel(const float* a, float s, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * s;
}

__global__ void addscale_kernel(const float* a, const float* b, float s,
                                float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = (a[i] + b[i]) * s;
}

static int mode_launch(const map<string, string>& m) {
  int iters = geti(m, "iters", 2000);
  chk(hipSetDevice(0));
  for (int i = 0; i < 20; i++) empty_kernel<<<1, 1>>>();
  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++) empty_kernel<<<1, 1>>>();
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double ms = event_ms(s, e);
  printf("{\"mode\":\"launch\",\"iters\":%d,\"total_ms\":%.4f,"
         "\"per_kernel_ms\":%.6f}\n",
         iters, ms, ms / iters);
  return 0;
}

static int mode_devices(const map<string, string>& m) {
  int nd = 0;
  chk(hipGetDeviceCount(&nd));
  printf("{\"mode\":\"devices\",\"count\":%d,\"devices\":[", nd);
  for (int d = 0; d < nd; d++) {
    hipDeviceProp_t prop;
    chk(hipGetDeviceProperties(&prop, d));
    printf("%s{\"id\":%d,\"name\":\"%s\",\"gcnArch\":\"%s\",\"major\":%d,\"minor\":%d}",
           d ? "," : "", d, prop.name, prop.gcnArchName, prop.major, prop.minor);
  }
  printf("]}\n");
  return 0;
}

static int mode_devkern(const map<string, string>& m) {
  // 诊断：在每张卡上启动一个自定义 kernel，定位 DTK 多设备模块错配
  int nd = 0;
  chk(hipGetDeviceCount(&nd));
  for (int d = 0; d < nd; d++) {
    chk(hipSetDevice(d));
    empty_kernel<<<1, 1>>>();
    chk(hipDeviceSynchronize());
    fprintf(stderr, "[devkern] device %d ok\n", d);
  }
  printf("{\"mode\":\"devkern\",\"ok\":true,\"count\":%d}\n", nd);
  return 0;
}

static int mode_gemm(const map<string, string>& m) {
  int dev = geti(m, "dev", 0);
  int M = geti(m, "m", 4096), N = geti(m, "n", 4096), K = geti(m, "k", 4096);
  int iters = geti(m, "iters", 10);
  chk(hipSetDevice(dev));

  rocblas_handle h;
  chk_r(rocblas_create_handle(&h));
  void *A = nullptr, *B = nullptr, *C = nullptr;
  chk(hipMalloc(&A, (size_t)M * K * 4));
  chk(hipMalloc(&B, (size_t)K * N * 4));
  chk(hipMalloc(&C, (size_t)M * N * 4));

  float alpha = 1.0f, beta = 0.0f;
  rocblas_operation op = rocblas_operation_none;
  for (int i = 0; i < 2; i++) {  // warmup
    chk_r(rocblas_sgemm(h, op, op, M, N, K, &alpha, (const float*)A, M,
                        (const float*)B, K, &beta, (float*)C, M));
  }
  chk(hipDeviceSynchronize());
  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++) {
    chk_r(rocblas_sgemm(h, op, op, M, N, K, &alpha, (const float*)A, M,
                        (const float*)B, K, &beta, (float*)C, M));
  }
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double ms = event_ms(s, e);
  double tflops = 2.0 * (double)M * N * K * iters / (ms * 1e-3) / 1e12;
  printf("{\"mode\":\"gemm\",\"m\":%d,\"n\":%d,\"k\":%d,\"dtype\":\"fp32\","
         "\"iters\":%d,\"ms\":%.4f,\"tflops\":%.4f}\n",
         M, N, K, iters, ms, tflops);
  return 0;
}

static int mode_copy(const map<string, string>& m) {
  int dev = geti(m, "dev", 0);
  size_t bytes = (size_t)getf(m, "bytes", 1e9);
  int iters = geti(m, "iters", 20);
  chk(hipSetDevice(dev));
  void *src = nullptr, *dst = nullptr;
  chk(hipMalloc(&src, bytes));
  chk(hipMalloc(&dst, bytes));
  chk(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++)
    chk(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double ms = event_ms(s, e);
  double gbps = (double)bytes * iters / (ms * 1e-3) / 1e9;
  printf("{\"mode\":\"copy\",\"bytes\":%zu,\"iters\":%d,\"ms\":%.4f,"
         "\"gbps\":%.4f}\n",
         bytes, iters, ms, gbps);
  return 0;
}

static int mode_triad(const map<string, string>& m) {
  // Stream triad：c[i] = a[i] + s*b[i]，测真实 kernel 的访存带宽（读2+写1）
  int dev = geti(m, "dev", 0);
  size_t n = (size_t)getf(m, "n", 1 << 26);  // 默认 2^26 元素（每个 4B）
  int iters = geti(m, "iters", 20);
  chk(hipSetDevice(dev));
  float *a = nullptr, *b = nullptr, *c = nullptr;
  chk(hipMalloc(&a, n * 4));
  chk(hipMalloc(&b, n * 4));
  chk(hipMalloc(&c, n * 4));
  int block = 256, grid = (int)((n + block - 1) / block);
  scale_kernel<<<grid, block>>>(a, 1.0f, c, (int)n);  // warmup
  chk(hipDeviceSynchronize());
  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++)
    addscale_kernel<<<grid, block>>>(a, b, 1.0f, c, (int)n);
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double ms = event_ms(s, e);
  double gbps = (double)n * 4 * 3 * iters / (ms * 1e-3) / 1e9;
  printf("{\"mode\":\"triad\",\"n\":%zu,\"iters\":%d,\"ms\":%.4f,"
         "\"gbps\":%.4f}\n",
         n, iters, ms, gbps);
  return 0;
}

static double d2d_copy(int dev_src, int dev_dst, size_t bytes, int iters) {
  void *a = nullptr, *b = nullptr;
  chk(hipSetDevice(dev_src));
  chk(hipMalloc(&a, bytes));
  chk(hipSetDevice(dev_dst));
  chk(hipMalloc(&b, bytes));
  chk(hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice));
  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++)
    chk(hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice));
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double ms = event_ms(s, e);
  hipFree(a);
  hipFree(b);
  return ms;
}

static int mode_pingpong(const map<string, string>& m) {
  int d0 = geti(m, "dev0", 0), d1 = geti(m, "dev1", 1);
  int latency = geti(m, "latency", 0);
  double bw_gbps = 0.0, lat_us = 0.0;
  if (latency) {
    size_t bytes = (size_t)getf(m, "bytes", 64);
    int iters = geti(m, "iters", 1000);
    double ms = d2d_copy(d0, d1, bytes, iters);
    lat_us = ms * 1e3 / (double)iters;  // 单方向（往返 / 2）
  } else {
    size_t bytes = (size_t)getf(m, "bytes", 2.56e8);
    int iters = geti(m, "iters", 20);
    double ms = d2d_copy(d0, d1, bytes, iters);
    bw_gbps = (double)bytes * iters / (ms * 1e-3) / 1e9;
  }
  printf("{\"mode\":\"pingpong\",\"dev0\":%d,\"dev1\":%d,"
         "\"latency\":%d,\"bw_gbps\":%.4f,\"latency_us\":%.4f}\n",
         d0, d1, latency, bw_gbps, lat_us);
  return 0;
}

static int mode_d2dasync(const map<string, string>& m) {
  // 诊断：从非 0 设备发起异步跨卡 D2D 拷贝（执行器里用的就是这个路径）
  int dev = geti(m, "dev", 3), peer = geti(m, "peer", 0);
  size_t bytes = (size_t)getf(m, "bytes", 1 << 20);
  int iters = geti(m, "iters", 5);
  void *a = nullptr, *b = nullptr;
  chk(hipSetDevice(dev));
  chk(hipMalloc(&a, bytes));
  chk(hipSetDevice(peer));
  chk(hipMalloc(&b, bytes));
  chk(hipSetDevice(dev));
  hipStream_t st;
  chk(hipStreamCreate(&st));
  hipEvent_t e;
  chk(hipEventCreate(&e));
  for (int i = 0; i < iters; i++)
    chk(hipMemcpyAsync(b, a, bytes, hipMemcpyDeviceToDevice, st));
  chk(hipEventRecord(e, st));
  chk(hipEventSynchronize(e));
  printf("{\"mode\":\"d2dasync\",\"dev\":%d,\"peer\":%d,\"ok\":true}\n", dev, peer);
  return 0;
}

static int mode_concurrency(const map<string, string>& m) {
  size_t bytes = (size_t)getf(m, "bytes", 2.56e8);
  int iters = geti(m, "iters", 20);

  void *a0, *b1, *a2, *b3;
  chk(hipSetDevice(0));
  chk(hipMalloc(&a0, bytes));
  chk(hipSetDevice(1));
  chk(hipMalloc(&b1, bytes));
  chk(hipSetDevice(2));
  chk(hipMalloc(&a2, bytes));
  chk(hipSetDevice(3));
  chk(hipMalloc(&b3, bytes));
  chk(hipMemcpy(b1, a0, bytes, hipMemcpyDeviceToDevice));
  chk(hipMemcpy(b3, a2, bytes, hipMemcpyDeviceToDevice));

  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  // 串行：先 pairA 后 pairB
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++)
    chk(hipMemcpy(b1, a0, bytes, hipMemcpyDeviceToDevice));
  for (int i = 0; i < iters; i++)
    chk(hipMemcpy(b3, a2, bytes, hipMemcpyDeviceToDevice));
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double t_seq_ms = event_ms(s, e);

  // 并发：两个线程各跑一对
  chk(hipEventRecord(s));
  std::thread ta([&] {
    chk(hipSetDevice(0));
    for (int i = 0; i < iters; i++)
      chk(hipMemcpy(b1, a0, bytes, hipMemcpyDeviceToDevice));
  });
  std::thread tb([&] {
    chk(hipSetDevice(2));
    for (int i = 0; i < iters; i++)
      chk(hipMemcpy(b3, a2, bytes, hipMemcpyDeviceToDevice));
  });
  ta.join();
  tb.join();
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double t_conc_ms = event_ms(s, e);

  double seq_gbps = 2.0 * (double)bytes * iters / (t_seq_ms * 1e-3) / 1e9;
  double conc_gbps = 2.0 * (double)bytes * iters / (t_conc_ms * 1e-3) / 1e9;
  double ratio = conc_gbps / seq_gbps;
  printf("{\"mode\":\"concurrency\",\"bytes\":%zu,\"iters\":%d,"
         "\"seq_ms\":%.4f,\"conc_ms\":%.4f,\"seq_gbps\":%.4f,"
         "\"conc_gbps\":%.4f,\"ratio\":%.4f}\n",
         bytes, iters, t_seq_ms, t_conc_ms, seq_gbps, conc_gbps, ratio);
  return 0;
}

static int mode_overlap(const map<string, string>& m) {
  int dev = geti(m, "dev", 0), peer = geti(m, "peer", 1);
  int M = geti(m, "m", 4096), N = geti(m, "n", 4096), K = geti(m, "k", 4096);
  size_t bytes = (size_t)getf(m, "bytes", 2.56e8);
  int iters = geti(m, "iters", 10);
  chk(hipSetDevice(dev));

  rocblas_handle h;
  chk_r(rocblas_create_handle(&h));
  void *A, *B, *C;
  chk(hipMalloc(&A, (size_t)M * K * 4));
  chk(hipMalloc(&B, (size_t)K * N * 4));
  chk(hipMalloc(&C, (size_t)M * N * 4));
  void *src = nullptr, *dst = nullptr;
  chk(hipMalloc(&src, bytes));
  chk(hipSetDevice(peer));
  chk(hipMalloc(&dst, bytes));
  chk(hipSetDevice(dev));

  hipStream_t gemm_stream, copy_stream;
  chk(hipStreamCreate(&gemm_stream));
  chk(hipStreamCreate(&copy_stream));
  chk_r(rocblas_set_stream(h, gemm_stream));

  float alpha = 1.0f, beta = 0.0f;
  rocblas_operation op = rocblas_operation_none;

  auto gemm_loop = [&] {
    for (int i = 0; i < iters; i++) {
      chk_r(rocblas_sgemm(h, op, op, M, N, K, &alpha, (const float*)A, M,
                          (const float*)B, K, &beta, (float*)C, M));
    }
  };

  hipEvent_t s, e, s2, e2;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));
  chk(hipEventCreate(&s2));
  chk(hipEventCreate(&e2));
  gemm_loop();  // warmup
  chk(hipStreamSynchronize(gemm_stream));
  chk(hipEventRecord(s, gemm_stream));
  gemm_loop();
  chk(hipEventRecord(e, gemm_stream));
  chk(hipStreamSynchronize(gemm_stream));
  double base_ms = event_ms(s, e);

  // 并发：拷贝在独立流上与 GEMM 真正重叠，只量 GEMM 段的降速（Echo 式）
  std::thread tc([&] {
    chk(hipSetDevice(dev));
    for (int i = 0; i < iters; i++)
      chk(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, copy_stream));
    chk(hipStreamSynchronize(copy_stream));
  });
  chk(hipEventRecord(s2, gemm_stream));
  gemm_loop();
  chk(hipEventRecord(e2, gemm_stream));
  tc.join();
  chk(hipStreamSynchronize(gemm_stream));
  double ovl_ms = event_ms(s2, e2);

  double ratio = ovl_ms / base_ms;
  printf("{\"mode\":\"overlap\",\"m\":%d,\"n\":%d,\"k\":%d,\"bytes\":%zu,"
         "\"base_ms\":%.4f,\"overlap_ms\":%.4f,\"ratio\":%.4f,"
         "\"interference\":%.4f}\n",
         M, N, K, bytes, base_ms, ovl_ms, ratio, ratio - 1.0);
  return 0;
}

static int mode_fusion(const map<string, string>& m) {
  int n = geti(m, "n", 1 << 24);
  int iters = geti(m, "iters", 100);
  chk(hipSetDevice(0));
  float *a, *b, *c, *d;
  chk(hipMalloc(&a, (size_t)n * 4));
  chk(hipMalloc(&b, (size_t)n * 4));
  chk(hipMalloc(&c, (size_t)n * 4));
  chk(hipMalloc(&d, (size_t)n * 4));
  int block = 256, grid = (n + block - 1) / block;

  add_kernel<<<grid, block>>>(a, b, c, n);
  scale_kernel<<<grid, block>>>(c, 2.0f, d, n);
  chk(hipDeviceSynchronize());
  hipEvent_t s, e;
  chk(hipEventCreate(&s));
  chk(hipEventCreate(&e));

  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++) {
    add_kernel<<<grid, block>>>(a, b, c, n);
    scale_kernel<<<grid, block>>>(c, 2.0f, d, n);
  }
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double separate_ms = event_ms(s, e);

  addscale_kernel<<<grid, block>>>(a, b, 2.0f, d, n);
  chk(hipDeviceSynchronize());
  chk(hipEventRecord(s));
  for (int i = 0; i < iters; i++)
    addscale_kernel<<<grid, block>>>(a, b, 2.0f, d, n);
  chk(hipEventRecord(e));
  chk(hipEventSynchronize(e));
  double fused_ms = event_ms(s, e);

  double per_extra_ms =
      fmax(0.0, fused_ms / iters - separate_ms / (2.0 * iters));
  printf("{\"mode\":\"fusion\",\"n\":%d,\"iters\":%d,"
         "\"separate_ms\":%.4f,\"fused_ms\":%.4f,"
         "\"per_extra_op_ms\":%.6f}\n",
         n, iters, separate_ms, fused_ms, per_extra_ms);
  return 0;
}

int main(int argc, char** argv) {
  map<string, string> m = parse_args(argc, argv);
  string mode = m.count("mode") ? m.at("mode") : "";
  if (mode == "launch") return mode_launch(m);
  if (mode == "devices") return mode_devices(m);
  if (mode == "devkern") return mode_devkern(m);
  if (mode == "gemm") return mode_gemm(m);
  if (mode == "copy") return mode_copy(m);
  if (mode == "triad") return mode_triad(m);
  if (mode == "pingpong") return mode_pingpong(m);
  if (mode == "d2dasync") return mode_d2dasync(m);
  if (mode == "concurrency") return mode_concurrency(m);
  if (mode == "overlap") return mode_overlap(m);
  if (mode == "fusion") return mode_fusion(m);
  die("unknown mode: " + mode);
  return 1;
}
