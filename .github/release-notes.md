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
