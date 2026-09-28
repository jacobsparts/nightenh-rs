//! PNG I/O and the two shape operations the pipeline needs.
//!
//! An image is `[c][h][w]` f32 in 0..1 - the same convention every engine in this
//! family uses - and the model's own preprocessing (`(x - 0.5) / 0.5`) is applied
//! where the arena is filled rather than here.
//!
//! Two shape operations, and no others:
//!
//! * [`resize`] - the bilinear resize to 512x512 that upstream's `predict.py`
//!   does before the network and the second one that puts the result back at the
//!   input's size afterwards. Half-pixel centres (`align_corners=False`), which
//!   is what `cv2.resize(..., INTER_LINEAR)` computes and therefore what the
//!   released tool does.
//! * [`pad_to_multiple`] - the generator is exactly shape-preserving only when
//!   both spatial dimensions are multiples of 4, and 4 is a hard requirement, not
//!   a preference: two strided convs with `ReflectionPad2d(1)` and k = 3 map
//!   n -> floor((n - 1)/2) + 1, so 33 -> 17 -> 9, and two exact-2x upsamples then
//!   return 36 rather than 33. Upstream never meets the case because it resizes to
//!   512 first. This engine's `--native` path pads each axis up to a multiple of
//!   `align` and crops back, which is the family's usual pad-and-crop convention
//!   and the reason `--factor` exists in a dev build.

use crate::Error;

#[derive(Clone)]
pub struct Image {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// `[c][h][w]`, each channel's plane contiguous.
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(c: usize, h: usize, w: usize) -> Image {
        Image { c, h, w, data: vec![0.0; c * h * w] }
    }

    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One channel's plane, as a slice.
    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.h * self.w;
        &self.data[c * n..(c + 1) * n]
    }
}

/// Read a PNG as `[3][h][w]` in 0..1. Palette and grayscale inputs are expanded to
/// RGB, matching `Image.convert('RGB')`, which is what upstream's loader does.
pub fn read_png(path: &str) -> Result<Image, Error> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    read_png_stream(std::io::BufReader::new(file), path)
}

/// The same, from any reader - which is what `-i -` needs.
pub fn read_png_stream<R: std::io::Read>(src: R, what: &str) -> Result<Image, Error> {
    let decoder = png::Decoder::new(src);
    let mut reader = decoder.read_info().map_err(|e| format!("{what}: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{what}: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let mut img = Image::new(3, h, w);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => {
            return Err(Error(format!("{what}: indexed PNG; re-save it as RGB")))
        }
    };
    for y in 0..h {
        for x in 0..w {
            let s = (y * w + x) * channels;
            let (r, g, b) = match channels {
                1 | 2 => (buf[s], buf[s], buf[s]),
                _ => (buf[s], buf[s + 1], buf[s + 2]),
            };
            for (i, v) in [r, g, b].iter().enumerate() {
                img.data[i * h * w + y * w + x] = *v as f32 / 255.0;
            }
        }
    }
    Ok(img)
}

/// Write `[3][h][w]` in 0..1 as an 8-bit PNG, rounding half up.
///
/// UPSTREAM TRUNCATES: `predict.py` saves with
/// `(np.clip(img, 0, 1) * 255.).astype(np.uint8)`, which drops the fraction, so
/// about half the pixels of a matched pair differ by one 8-bit level and the two
/// pictures agree to roughly 50 dB rather than exactly. Rounding is kept because
/// it is what the rest of this family does and what the numpy reference does, so
/// the engine and its reference agree exactly; the difference against upstream's
/// own PNGs is half a level on average, far below the model's own error.
pub fn write_png(path: &str, img: &Image) -> Result<(), Error> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    write_png_stream(std::io::BufWriter::new(file), img, path)
}

/// The same, to any writer - `-o -` writes the PNG to stdout.
pub fn write_png_stream<W: std::io::Write>(dst: W, img: &Image, what: &str) -> Result<(), Error> {
    let mut encoder = png::Encoder::new(dst, img.w as u32, img.h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| format!("{what}: {e}"))?;
    let mut out = vec![0u8; 3 * img.h * img.w];
    for y in 0..img.h {
        for x in 0..img.w {
            for c in 0..3 {
                let v = img.data[c * img.h * img.w + y * img.w + x].clamp(0.0, 1.0);
                out[(y * img.w + x) * 3 + c] = (v * 255.0 + 0.5) as u8;
            }
        }
    }
    writer.write_image_data(&out).map_err(|e| format!("{what}: {e}"))?;
    Ok(())
}

/// Bilinear resize to `(h, w)`, half-pixel centres: `src = (dst + 0.5) * scale -
/// 0.5`, then a linear blend of the two nearest samples with the edges clamped.
///
/// This is `cv2.resize(..., INTER_LINEAR)` and torch's
/// `interpolate(mode='bilinear', align_corners=False)`, which is what upstream's
/// pipeline uses in both directions. Separable, so one pass per axis: for a
/// 512x512 target from a 4K input the horizontal pass is the same work either way
/// and the vertical pass then reads a much smaller tensor.
pub fn resize(img: &Image, h: usize, w: usize) -> Image {
    if img.h == h && img.w == w {
        return Image { c: img.c, h, w, data: img.data.clone() };
    }
    let plane = img.h * img.w;
    let mut tmp = vec![0.0f32; img.c * img.h * w];
    let xs: Vec<f32> = (0..w).map(|j| ((j as f32 + 0.5) * img.w as f32 / w as f32) - 0.5).collect();
    for c in 0..img.c {
        let src = &img.data[c * plane..(c + 1) * plane];
        let dst = &mut tmp[c * img.h * w..(c + 1) * img.h * w];
        for y in 0..img.h {
            let row = &src[y * img.w..(y + 1) * img.w];
            let out = &mut dst[y * w..(y + 1) * w];
            for (j, &x) in xs.iter().enumerate() {
                let x0 = x.floor();
                let t = x - x0;
                let i0 = (x0 as isize).clamp(0, img.w as isize - 1) as usize;
                let i1 = (i0 as isize + 1).clamp(0, img.w as isize - 1) as usize;
                out[j] = row[i0] * (1.0 - t) + row[i1] * t;
            }
        }
    }
    let ys: Vec<f32> = (0..h).map(|i| ((i as f32 + 0.5) * img.h as f32 / h as f32) - 0.5).collect();
    let mut out = Image::new(img.c, h, w);
    let tplane = img.h * w;
    for c in 0..img.c {
        let src = &tmp[c * tplane..(c + 1) * tplane];
        let dst = &mut out.data[c * h * w..(c + 1) * h * w];
        for (i, &y) in ys.iter().enumerate() {
            let y0 = y.floor();
            let t = y - y0;
            let r0 = (y0 as isize).clamp(0, img.h as isize - 1) as usize;
            let r1 = (r0 as isize + 1).clamp(0, img.h as isize - 1) as usize;
            for j in 0..w {
                dst[i * w + j] = src[r0 * w + j] * (1.0 - t) + src[r1 * w + j] * t;
            }
        }
    }
    out
}

/// Pad each spatial axis UP to a multiple of `align` with a REFLECTION, and
/// return the padding so the result can be cropped back.
///
/// The generator needs each dimension to be a multiple of 4 (see the module note)
/// and this engine pads to 64 by default, which is a multiple of 4 and also what
/// makes the 7x7 tiles and the channel-mean blocks land on whole rows. Reflection
/// is the same rule the network's own padding uses, which matters: padding with
/// zeros or with the edge value would put a different border into the model at a
/// size where the border is 1/8th of the picture.
///
/// Reflection needs `pad < dim`, so a dimension smaller than the pad is a
/// refusal rather than a silent clamp - at that size the model has no defined
/// behaviour anyway.
pub fn pad_to_multiple(img: &Image, align: usize) -> Result<(Image, [usize; 4]), Error> {
    let ph = (align - img.h % align) % align;
    let pw = (align - img.w % align) % align;
    let (top, left) = (ph / 2, pw / 2);
    let (bottom, right) = (ph - top, pw - left);
    if ph == 0 && pw == 0 {
        return Ok((Image { c: img.c, h: img.h, w: img.w, data: img.data.clone() },
                   [0, 0, 0, 0]));
    }
    if img.h <= top.max(bottom) || img.w <= left.max(right) {
        return Err(Error(format!(
            "{}x{} is too small to reflection-pad to a multiple of {align}",
            img.w, img.h
        )));
    }
    Ok((reflect_pad2(img, top, bottom, left, right), [top, bottom, left, right]))
}

/// `jnp.pad(..., mode='reflect')` on the two spatial axes, `before`/`after` per
/// axis. Reflection EXCLUDES the edge sample, so index -1 of the padded axis is
/// the second sample of the input; `symmetric` padding (which repeats the edge)
/// is a different function.
pub fn reflect_pad2(img: &Image, top: usize, bottom: usize, left: usize, right: usize) -> Image {
    let (h, w) = (img.h, img.w);
    let (nh, nw) = (h + top + bottom, w + left + right);
    let refl = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let mut i = i;
        while i < 0 || i >= n {
            i = if i < 0 { -i } else { 2 * n - 2 - i };
        }
        i as usize
    };
    let mut out = Image::new(img.c, nh, nw);
    for c in 0..img.c {
        let src = img.plane(c);
        let dst = &mut out.data[c * nh * nw..(c + 1) * nh * nw];
        for oy in 0..nh {
            let sy = refl(oy as isize - top as isize, h);
            for ox in 0..nw {
                let sx = refl(ox as isize - left as isize, w);
                dst[oy * nw + ox] = src[sy * w + sx];
            }
        }
    }
    out
}

/// Crop `[top, bottom, left, right]` back out of a padded image.
pub fn crop(img: &Image, pad: [usize; 4]) -> Image {
    let [top, _bottom, left, _right] = pad;
    let mut out = Image::new(img.c, img.h - pad[0] - pad[1], img.w - pad[2] - pad[3]);
    let hw = img.h * img.w;
    for c in 0..img.c {
        for y in 0..out.h {
            let sy = y + top;
            let s = c * hw + sy * img.w + left;
            let d = c * out.h * out.w + y * out.w;
            out.data[d..d + out.w].copy_from_slice(&img.data[s..s + out.w]);
        }
    }
    out
}
