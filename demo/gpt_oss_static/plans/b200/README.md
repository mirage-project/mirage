Plan files for the 148-SM B200: `<kind>_e<experts>_r<requests>[_t<rows>].json` (`plan_path` in `gpt_oss_static.py`;
kind `sliding`, `full` or `head`), made by the same `--tune` pipeline as the H100 plans (see `../h100/README.md`), with
FlashInfer's trtllm-gen MXFP4 MoE outside the layer kernel. Each is the faster of round robin / the solver's plan on the
layer span (timing build, L2 emptied); e.g. full attention at 2 requests 52.5 -> 45.7 us, sliding at 64 requests (K
split floor) 57.9 -> 48.4 us.
