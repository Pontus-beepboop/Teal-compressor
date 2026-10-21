# Contributing

Thanks for helping. Bug reports, test results and pull requests are all welcome.

## Reporting results and bugs

The most useful reports say:

* the camera, firmware and recording settings (resolution, frame rate, N-RAW quality);
* the Blue Compressor version, OS, and backend (`bluecompress backends`);
* the exact command or app settings, and what went wrong;
* if you can, a few seconds of sample footage that shows the problem. Clips are large,
  so please link to them rather than attaching them.

Reports from cameras, operating systems and GPUs not tested yet (Windows, Linux and
CUDA in particular) are very welcome, even when everything works.

## Pull requests

1. Open an issue first for anything larger than a small fix, so we can agree on the
   approach.
2. Keep each change focused. Run these before pushing:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets --release
   cargo test --release -p bluecompress
   ```
3. Changes to the codec, container writer or resampler must keep **Q1 lossless**.
   Run the clip tests on real footage (`BC_TEST_CLIPS`, see the README) and say in the
   pull request what you tested.
4. Never commit camera footage, or code, headers, libraries or documentation from any
   vendor SDK, and do not contribute anything learned from a vendor SDK or other
   material under a license that forbids reverse engineering. The `.gitignore` excludes the usual file types; please keep it that way.

By contributing you agree that your contributions are licensed under the Apache License
2.0, as stated in section 5 of the license.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
