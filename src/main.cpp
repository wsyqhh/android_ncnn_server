// apex_server — NCNN 原生推理服务器 (运行在安卓平板 Termux)
// 通过 TCP 接收 BGR 图像，NCNN+Vulkan GPU 推理，回传检测结果
#include "net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

static const uint32_t MAGIC_REQUEST  = 0xAA550001;
static const uint32_t MAGIC_RESPONSE = 0xBB660001;

struct Detection {
    int32_t x, y, w, h;
    float confidence;
    int32_t class_id;
};

// === 工具函数 ===
static bool recv_exact(int fd, void* buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = recv(fd, (char*)buf + done, n - done, 0);
        if (r <= 0) return false;
        done += r;
    }
    return true;
}

static float compute_iou(const Detection& a, const Detection& b) {
    float ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx2 = b.x + b.w, by2 = b.y + b.h;
    float ix1 = std::max((float)a.x, (float)b.x), iy1 = std::max((float)a.y, (float)b.y);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.f, ix2-ix1), ih = std::max(0.f, iy2-iy1);
    float inter = iw * ih;
    return inter / (a.w*a.h + b.w*b.h - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thr) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<bool> sup(dets.size(), false);
    std::vector<Detection> res;
    for (size_t i = 0; i < dets.size(); i++) {
        if (sup[i]) continue;
        res.push_back(dets[i]);
        for (size_t j = i+1; j < dets.size(); j++)
            if (!sup[j] && compute_iou(dets[i], dets[j]) > iou_thr) sup[j] = true;
    }
    return res;
}

// === NCNN 推理器 ===
class NcnnDetector {
public:
    ncnn::Net net;
    int target_size;
    float conf_thr, iou_thr;

    NcnnDetector() : target_size(640), conf_thr(0.60f), iou_thr(0.45f) {}

    bool load(const char* param, const char* bin, bool gpu) {
        net.opt.use_vulkan_compute = gpu;
        net.opt.num_threads = 4;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        if (net.load_param(param) || net.load_model(bin)) {
            fprintf(stderr, "[错误] 模型加载失败\n");
            return false;
        }
        fprintf(stderr, "[NCNN] 模型加载成功 (GPU: %s)\n", gpu ? "Vulkan" : "CPU");
        return true;
    }

    std::vector<Detection> detect(const unsigned char* bgr, int iw, int ih) {
        float scale = std::min((float)target_size/iw, (float)target_size/ih);
        int nw = (int)(iw*scale), nh = (int)(ih*scale);
        int pw = (target_size-nw)/2, ph = (target_size-nh)/2;

        ncnn::Mat in = ncnn::Mat::from_pixels(bgr, ncnn::Mat::PIXEL_BGR2RGB, iw, ih);
        ncnn::Mat resized;
        ncnn::resize_bilinear(in, resized, nw, nh);
        ncnn::Mat padded;
        ncnn::copy_make_border(resized, padded, ph, target_size-nh-ph,
                               pw, target_size-nw-pw, ncnn::BORDER_CONSTANT, 114.f);
        const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
        padded.substract_mean_normalize(0, norm);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("images", padded);
        ncnn::Mat out;
        ex.extract("output0", out);

        // YOLOv8 后处理
        int rows = out.h, cols = out.w;
        bool tr = rows < cols;
        int np = tr ? cols : rows, nf = tr ? rows : cols;
        int nc = nf - 4;
        if (nc <= 0) return {};

        std::vector<Detection> raw;
        for (int i = 0; i < np; i++) {
            auto g = [&](int f) -> float { return tr ? out.row(f)[i] : out.row(i)[f]; };
            float ms = 0; int mc = 0;
            for (int c = 0; c < nc; c++) { float s = g(4+c); if (s > ms) { ms=s; mc=c; } }
            if (ms <= conf_thr) continue;
            float cx=g(0),cy=g(1),bw=g(2),bh=g(3);
            raw.push_back({(int)((cx-bw/2-pw)/scale), (int)((cy-bh/2-ph)/scale),
                           (int)(bw/scale), (int)(bh/scale), ms, mc});
        }
        return nms(raw, iou_thr);
    }
};

// === TCP 服务器 ===
int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    int port = 10000;
    int input_size = 640;
    float conf = 0.60f;
    const char* param_file = "apex_final.param";
    const char* bin_file = "apex_final.bin";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--port") && i+1<argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--size") && i+1<argc) input_size = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--conf") && i+1<argc) conf = atof(argv[++i]);
        else if (!strcmp(argv[i],"--param") && i+1<argc) param_file = argv[++i];
        else if (!strcmp(argv[i],"--bin") && i+1<argc) bin_file = argv[++i];
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    bool use_gpu = ncnn::get_gpu_count() > 0;
    if (use_gpu) fprintf(stderr, "[GPU] 检测到 %d 个 GPU (Vulkan)\n", ncnn::get_gpu_count());
#else
    bool use_gpu = false;
#endif

    NcnnDetector det;
    det.target_size = input_size;
    det.conf_thr = conf;
    if (!det.load(param_file, bin_file, use_gpu)) return 1;

    // 预热
    fprintf(stderr, "[预热] 运行 3 次推理...\n");
    std::vector<unsigned char> fake(416*416*3, 128);
    for (int i = 0; i < 3; i++) det.detect(fake.data(), 416, 416);

    // 基准测试
    struct timespec t0, t1;
    double total = 0;
    for (int i = 0; i < 20; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        det.detect(fake.data(), 416, 416);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    }
    fprintf(stderr, "[基准] 平均推理: %.1fms | FPS: %.0f\n", total/20, 20000.0/total);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);

    fprintf(stderr, "\n=== 推理服务器已启动 端口:%d ===\n等待 PC 连接...\n\n", port);

    while (true) {
        int cli = accept(srv, NULL, NULL);
        setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
        fprintf(stderr, "[连接] PC 已连接\n");

        uint64_t fc = 0; double tt = 0;
        uint8_t hdr[8];
        while (recv_exact(cli, hdr, 8)) {
            uint32_t mag = *(uint32_t*)hdr, sz = *(uint32_t*)(hdr+4);
            if (mag != MAGIC_REQUEST) break;

            std::vector<uint8_t> img(sz);
            if (!recv_exact(cli, img.data(), sz)) break;

            int crop = (int)sqrt(sz / 3);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            auto dets = det.detect(img.data(), crop, crop);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            tt += (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
            fc++;

            uint32_t cnt = std::min((int)dets.size(), 64);
            uint32_t resp[2] = {MAGIC_RESPONSE, cnt};
            send(cli, resp, 8, 0);
            if (cnt > 0) send(cli, dets.data(), cnt*24, 0);

            if (fc % 200 == 0)
                fprintf(stderr, "[性能] 推理:%.1fms FPS:%.0f 帧:%lu\n", tt/fc, fc*1000.0/tt, fc);
        }
        close(cli);
        fprintf(stderr, "[断开] 共 %lu 帧\n等待重连...\n", fc);
    }
}
