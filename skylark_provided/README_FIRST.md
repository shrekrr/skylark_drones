# Skylark GCP Inference Pipeline Assignment — package guide

Package version: **1.0**

Assignment page: **[Open the Notion assignment](https://somber-freezer-f15.notion.site/Skylark-GCP-Inference-Pipeline-Assignment-39d373981d1480629cbdebc5ec8785a1)**

This download contains development and test GeoTIFFs, development annotations, the frozen ONNX model and its public specification, JSON schemas, and a minimal container starter. The complete problem statement is hosted separately.

Total package size: **9.81 GiB (10,534,065,069 bytes)**

Raster sizes:

- `dev_001.tif`: 3.41 GiB (3,663,360,194 bytes)
- `dev_002.tif`: 163.03 MiB (170,945,936 bytes)
- `dev_003.tif`: 1.36 GiB (1,465,289,797 bytes)
- `dev_004.tif`: 635.33 MiB (666,192,985 bytes)
- `test_001.tif`: 3.43 GiB (3,688,027,744 bytes)
- `test_002.tif`: 739.38 MiB (775,299,963 bytes)

Verify all files from this directory with:

```bash
sha256sum --check checksums.sha256
```

Your solution must inspect each GeoTIFF's embedded metadata, including dimensions, bands, dtype, CRS, affine transform, mask/nodata semantics, and resolution. Do not infer these properties from filenames.

The model and data are provided only for this hiring exercise and must not be redistributed.
