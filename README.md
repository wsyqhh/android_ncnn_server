# Apex NCNN Android Server

安卓平板原生 NCNN 推理服务器，通过 USB TCP 接收 PC 截屏图像，Vulkan GPU 加速推理。

## 使用方法

1. Fork 或推送此仓库到你的 GitHub
2. 等待 Actions 自动编译（约5分钟）
3. 在 Actions 页面下载 `apex-server-android-arm64` 产物
4. 解压得到 `apex_server` + `apex_final.param` + `apex_final.bin`
5. 推送到平板：
```bash
adb push apex_server /data/data/com.termux/files/home/
adb push apex_final.param /data/data/com.termux/files/home/
adb push apex_final.bin /data/data/com.termux/files/home/
```
6. 在 Termux 中运行：
```bash
chmod +x ~/apex_server
~/apex_server
```
