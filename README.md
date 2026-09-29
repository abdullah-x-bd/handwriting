# Hand Magic PDF

This repository runs the real Hand Magic neural handwriting generator and composes its generated strokes into A4 PDF pages.

## Default preset for future use

**Use the readable full-page preset for future handwritten PDF generation.**

The approved reference is the configuration that produced `HandMagic_A1_Legibility_Fixed`.

Core defaults:

- Primary Hand Magic writer: `2808`
- Fallback writer: `1021`
- Bias: `24.0`
- Balanced line target: `55` characters
- Balanced line maximum: `62` characters
- A4 raster: `2480 x 3508`
- Preserve handwriting aspect ratio: **yes**
- Natural writing-width target: about **95%**
- Reject collapsed / illegible generations: **yes**
- Dynamic vertical page fill: **yes**

Most important rule: **do not horizontally stretch the handwriting to fill the page**. Generate longer balanced lines and scale them uniformly so the letter shapes stay readable.

Full settings and quality-control rules:

- `HANDMAGIC_FUTURE_REFERENCE.md`
- `HANDMAGIC_FUTURE_REFERENCE.json`

Executable reference implementation:

- `readable_fullpage_sample.py`

Treat those files as the source of truth for future Hand Magic work.
