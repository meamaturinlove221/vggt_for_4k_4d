# DNA 4K4D Notes

## Current local finding

The current local mirror under `G:\数据集\datasets` is not a complete `data_used_in_4K4D` bundle.

- Present zip parts: `001`, `003`, `008` to `017`
- Missing zip parts by filename gap: `002`, `004`, `005`, `006`, `007`
- The embedded `data_used_in_4K4D_file_gid.json` confirms that `data_used_in_4K4D/main/*.smc` files are expected.
- The embedded `data_used_in_4K4D_rgb_cams.zip` only provides `*_rgb_cams.smc` camera-parameter files.

That means the current local copy can support camera-parameter inspection, but it cannot support the first real RGB bridge step until the missing `main/*.smc` files are present.

## Commands

Inventory the current local mirror:

```powershell
python tools/dna_4k4d.py inventory `
  --dataset-path "G:\数据集\datasets" `
  --json-out "D:\vggt\vggt-main\reports\dna_4k4d_inventory.json"
```

Assemble all currently available outer zip parts:

```powershell
python tools/dna_4k4d.py assemble `
  --dataset-path "G:\数据集\datasets" `
  --target-root "G:\数据集\datasets\assembled_4k4d" `
  --skip-existing `
  --extract-inner-rgb-cams
```

Build a one-sequence manifest:

```powershell
python tools/dna_4k4d.py manifest `
  --dataset-path "G:\数据集\datasets" `
  --seq 0012_11 `
  --frame 0 `
  --target-camera 00 `
  --auto-sources 6 `
  --allow-partial `
  --output-dir "D:\vggt\vggt-main\reports\dna_case_probe"
```

If the missing `main/*.smc` files are later downloaded but still kept inside zip parts, probe them on demand:

```powershell
python tools/dna_4k4d.py manifest `
  --dataset-path "G:\数据集\datasets" `
  --seq 0012_11 `
  --frame 0 `
  --target-camera 00 `
  --auto-sources 6 `
  --materialize-archived `
  --allow-partial `
  --output-dir "D:\vggt\vggt-main\reports\dna_case_probe"
```

Export one 4K4D frame as a plain scene folder for the official VGGT inference path:

```powershell
python tools/export_4k4d_scene.py `
  --dataset-root "G:\数据集\datasets\data_used_in_4K4D" `
  --seq 0012_11 `
  --frame 0 `
  --target-camera 00 `
  --auto-sources 6 `
  --output-dir "D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_7views" `
  --overwrite
```

Run the exported scene through official VGGT on Modal:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_4k4d_vggt_infer.ps1 `
  -LocalSceneDir "D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_7views" `
  -OutputSubdir "vggt_4k4d_infer/0012_11_frame0000_7views"
```

Pull the cloud outputs back to local disk:

```powershell
modal volume get vggt-4k4d-output `
  /vggt_4k4d_infer/0012_11_frame0000_7views `
  "D:\vggt\vggt-main\output\modal_results"
```
