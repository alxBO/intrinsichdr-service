# IntrinsicHDR Service

Web service for single-image LDR-to-HDR reconstruction using **IntrinsicHDR** (ECCV 2024).

The model decomposes an LDR image into intrinsic components (albedo, shading, illumination) and uses these to reconstruct physically plausible HDR content, recovering highlight detail and expanding dynamic range beyond what simple inverse tone mapping can achieve.

Paper: *IntrinsicHDR: Single Image Reverse Tone Mapping via Intrinsic Decomposition* (ECCV 2024)

## Quick Start

### Mac (MPS / CPU)

```bash
pip install -r service/backend/requirements.txt
./service/run_mac.sh
# Open http://localhost:8004
```

Models (~500 MB) auto-download via `torch.hub` on first run.

### Vast.ai (GPU)

```bash
git clone --recurse-submodules <repo-url>
cd intrinsichdr-service/service
./deploy_vastai.sh
```

## Features

- Single-image LDR to HDR conversion
- Client-side tone mapping preview (ACES / Reinhard / Linear)
- A/B comparison slider (SDR input vs HDR output)
- EXR download for full-precision HDR output
- Luminance analysis: dynamic range, percentiles, contrast metrics
- Input/output histogram visualization

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Max Resolution | 256 - 8192 | 4096 | Longest side limit before processing |
| Image Scale | 0.1 - 5.0 | 1.0 | Scale factor applied to input image |
| Processing Scale | 0.25 - 2.0 | 1.0 | Internal processing resolution multiplier |

## Limitations

- Single-image method: no multi-exposure bracketing, so extreme highlights rely on learned priors
- Processing time scales with resolution; high-res images (>4K) may need significant VRAM
- Best results on natural photographs with visible highlight regions
