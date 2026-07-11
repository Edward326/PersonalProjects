# bmp_colorTRANSFORMER

## Purpose

A command-line C utility that applies color filters directly to uncompressed **BMP** image
files. It takes one or more `.bmp` files as arguments, lets the user choose a filter
(black & white or sepia), and writes out a filtered copy of each image
(`<name>_mod.bmp`) next to the original, without relying on any external image library —
the BMP file format is parsed and rewritten by hand, pixel by pixel.

## Implementation

### BMP parsing
The `image` struct mirrors the BMP file layout field by field (signature, file size,
header size, width, height, planes, bits-per-pixel, compression, image size, pixels-per-meter,
color table sizes) plus a dynamically allocated 2D pixel array:

```c
typedef struct{
  char sign[2];
  int file_size, rez1, offset_start;
  int sizeH, width, height, planes, btc_px, comprss, img_size;
  int px_x, px_y, size_color, size_icolor;
  int **array;
} image;
```

- **`save()`** reads a `.bmp` file field-by-field with `fread`, validates the `"BM"` signature,
  and loads the pixel grid into a heap-allocated `int**` array (one row per scanline).
- **`load()`** performs the inverse operation with `fwrite`, writing the (possibly modified)
  header and pixel data back out to a new file.
- Both 24-bit and 32-bit-per-pixel BMPs are supported; pixel values are packed/unpacked with bit
  shifting and masking (e.g. `(nr >> 16) & 255` to isolate the red channel of a 24-bit pixel).

### Color filters
Two filters are implemented as pure functions operating on a single packed pixel value:

- **Black & white** (`black_white24B` / `black_white32B`) — averages the red, green and blue
  channels and writes that average back into all three channels, per pixel.
- **Sepia** (`sepia_24B` / `sepia_32B`) — applies the standard sepia color transform matrix
  (`0.393R + 0.769G + 0.189B`, etc. for each channel), clamping any overflow to the maximum
  channel value (255 for 24-bit, 511 for 32-bit).

`bw_save()` and `sepia_save()` walk every pixel of an `image` and apply the corresponding filter
in place, branching on the detected bit depth (24 vs 32 bits per pixel).

### CLI / program flow (`main`)
1. Reads the terminal size via `ioctl(TIOCGWINSZ)` for later screen-clearing/formatting.
2. Accepts any number of BMP filenames as command-line arguments (`./exec img1.bmp img2.bmp ...`).
3. A special `help` argument prints usage instructions instead of running (`help()`).
4. Each valid, openable BMP file is loaded into memory via `save()`; a matching output filename
   (`original_mod.bmp`) is built by string manipulation for each successfully loaded image.
5. The user is prompted with a menu (`options()`) to pick **1 = black & white** or
   **2 = sepia**.
6. The chosen filter is applied to every loaded image, and each result is written out through
   `load()`.
7. `free_up1()` / `free_up2()` release the dynamically allocated image and filename arrays
   before the program exits.

### Notes
- The program is Linux-specific (`sys/ioctl.h`, `TIOCGWINSZ`) since it uses the terminal window
  size purely for cosmetic spacing when printing status/help text.
- The repository includes several sample BMP files (`sample.bmp`, `stonks.bmp`, `r8.bmp`) together
  with their already-filtered `_mod` counterparts, useful as before/after references.
- The `test.c` file is an earlier, single-image version of the program; `test_multipleV2.0.c`
  (the one described above) is the current, multi-file version with fixed memory management
  ("v2.0A(with no memory leaks)" per its own startup banner).
