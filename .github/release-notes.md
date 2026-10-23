**What's new in 0.1.3**
- AMD (OpenCL) and NVIDIA (CUDA) GPU acceleration is now enabled in both the Windows and Linux desktop app and the command line tool when downsizing to 6K or 4K, with automatic fallback to the CPU when no supported GPU or runtime is present. Both GPU paths now use a double-precision transfer curve and disable FMA contraction where the CPU does, matching CPU output bit for bit.
- Validated CUDA support on Windows 11 (RTX 4060 Ti), clearer error reporting and documentation for the NVRTC 12.x runtime requirement on Windows, and a CUDA kernel compile and load test (thanks to @Lucke-bp in PR #2).

**What's new in 0.1.2**
- Experimental GPU acceleration on Windows and Linux, in both the app and the command line tool: AMD graphics cards through OpenCL and Nvidia graphics cards through CUDA. The GPU is only used when downsizing to 6K or 4K; at the original resolution everything runs on the CPU as before. Without a supported GPU and driver it falls back to the CPU automatically.
- This is untested on real AMD and Nvidia hardware, so feedback is very welcome. If you have one of these cards:
  1. Check that the engine label in the app (or `bluecompress backends`) shows your graphics card.
  2. Compress a clip to 6K or 4K once with GPU and once with CPU at the same quality, and compare the two output files' hashes (they should be identical).
  3. Report the speed of both runs, and any crashes or errors, with your card and driver version.

**What's new in 0.1.1**
- Fixed: queue rows no longer get squashed when a large batch of clips is added (for example a few hundred). Rows keep their full height and the list scrolls, so you can remove or change settings for any clip.

Early, experimental build. Tested on one Nikon Z8 (N-RAW) clip and one Nikon ZR (R3D NE) clip on macOS (Apple silicon). Windows and Linux builds and the CUDA backend are untested. Keep your originals until you have checked the outputs.

**Downloads**
- macOS: `.dmg` (Apple silicon `arm64` or Intel `x64`)
- Windows: `-setup.exe` installer or `.msi`
- Linux: `.AppImage`, `.deb` or `.rpm`
- Command line tool, library and C header: `bluecompress-cli_*` archives

**Unsigned builds.** These builds are not signed with an Apple or Microsoft developer certificate.
- **macOS**: drag the app to Applications and open it. macOS says it cannot verify the app; click **Done**, then open **System Settings > Privacy & Security**, scroll down and click **Open Anyway**. You only need to do this once. On macOS 14 and earlier, right-clicking the app and choosing **Open** also works.
- **Windows**: SmartScreen may warn; choose **More info**, then **Run anyway**.
