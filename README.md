<!-- markdownlint-disable no-inline-html no-emphasis-as-heading first-line-h1 no-duplicate-heading -->
# Morphace

<div align="center">

![Morphace Banner](https://morphace.daspyro.de/media/morphace_banner.png)
**Generate a face-morphing video from two face images.**

</div>

[Morphace](https://morphace.daspyro.de/) detects facial landmarks, aligns two faces, blends between them,
and exports the result as an MP4 video.

## Features

- Generate smooth MP4 face-morphing videos from two face images.
- Align and crop portraits before morphing for better results.
- Batch-process folders of images with automatic face detection.
- Customize video length, frame rate, and output location.

## Video Example

[![Morph Example Image](https://morphace.daspyro.de/media/img_gallery_link.png)](https://morphace.daspyro.de)

<!-- TABLE OF CONTENTS -->
<details closed="closed">
<summary><h2 style="display: inline-block">Table of Contents</h2></summary>

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Morphace CLI](#morphace-cli)
  - [download](#download)
  - [prep](#prep)
  - [morph](#morph)
- [Tips for Best Results](#tips-for-best-results)
- [License](#license)

</details>

## Requirements

Morphace requires:

- Python 3.12 or newer.
- FFmpeg installed and available on your system `PATH`.

The required Python packages are installed automatically when you install
Morphace.

You can verify that FFmpeg is available by running:

```bash
ffmpeg -version
```

**Face landmark model**

Morphace also requires a face landmark model file to detect and align faces.
The model is not bundled with Morphace, so download it once after installing:

```bash
morphace download
```

By default, the command saves the model in Morphace's
[application data directory](https://morphace.daspyro.de/faq/model-file-location/).
The [`prep`](https://morphace.daspyro.de/cli-reference/prep/) and [`morph`](https://morphace.daspyro.de/cli-reference/morph/) commands look there automatically.

Use `--landmark-model` with `morphace prep` or `morphace morph` only if you
want to [use a model file saved somewhere else](https://morphace.daspyro.de/faq/alt-model-location/).

## Quick Start

1. Install the Morphace package with pip:

    ```bash
    pip install morphace
    ```

2. Download the required face landmark model.

    ```bash
    morphace download
    ```

3. Create a morph video:

    ```bash
    morphace morph first_image.png second_image.png
    ```

4. Open the generated video file `morph.mp4`.

## How It Works

Morphace detects landmarks in both images, aligns the faces to a shared shape,
warps each frame between the two faces, blends the image content, and encodes
the result as an MP4 video.

At a high level, Morphace:

1. Detects a face in each source image.
2. Finds facial landmarks such as the eyes, nose, mouth, and face outline.
3. Aligns both faces to a shared shape.
4. Warps and blends the images frame by frame.
5. Encodes the generated frames into an MP4 video.

For a more detailed explanation of the landmark detection,
triangle mesh, and frame interpolation process, refer to
[How It Works](https://morphace.daspyro.de/how-it-works/) on the Morphace
documentation site.

## Morphace CLI

The morphace command provides three subcommands:

| Command                                                                    | Purpose                                                      |
| -------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [`morphace download`](https://morphace.daspyro.de/cli-reference/download/) | Downloads Morphace's required model file.                    |
| [`morphace prep`](https://morphace.daspyro.de/cli-reference/prep/)         | Prepares aligned square PNG face crops from raw images.      |
| [`morphace morph`](https://morphace.daspyro.de/cli-reference/morph/)       | Generates an MP4 face-morphing video from two source images. |

Each of these commands is explained briefly below. For full details, refer
to [CLI Reference](https://morphace.daspyro.de/cli-reference/) on the
Morphace documentation site.

### download

The [`download`](https://morphace.daspyro.de/cli-reference/download/) command installs the model file described in
[Requirements](https://morphace.daspyro.de/install/#requirements).

```bash
morphace download
```

By default, the model is saved in Morphace's
[application data directory](https://morphace.daspyro.de/faq/model-file-location/),
where `morphace prep` and `morphace morph` can find it automatically. You
usually only need to run this command once after installing Morphace.

To save the model somewhere else, use `--save-to`:

```bash
morphace download --save-to ./model
```

If a model file already exists, Morphace will not overwrite it unless you
pass `--force`:

```bash
morphace download --force
```

Use `--debug` to show more detailed logging if the download fails.

For full details, refer to
[download CLI Reference](https://morphace.daspyro.de/cli-reference/download/) on the
Morphace documentation site.

### prep

Use [`morphace prep`](https://morphace.daspyro.de/cli-reference/prep/) to turn raw portraits into aligned square PNG face crops
before creating a morph.

This step is optional, but recommended when your source images are framed
differently. For example, a close-up portrait and a full-body photo may both
contain detectable faces, but the morph will usually look better if both faces
are aligned and cropped first.

```bash
morphace prep images/raw images/aligned
```

The [`prep`](https://morphace.daspyro.de/cli-reference/prep/) command can process either a single image file or a directory of
images. When processing a directory, Morphace scans the immediate child files
with these extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, and `.webp`.

For each detected face, Morphace writes one aligned PNG file. Output files are
named with the original file stem and face number, such as:

```text
portrait_face01.png
portrait_face02.png
```

By default, aligned crops are saved as 1024x1024 PNG images.

#### Examples

Prepare a directory of images:

```bash
morphace prep \
  images/raw \
  images/aligned
```

Prepare a single image and save the output beside the original file:

```bash
morphace prep images/raw/portrait.jpg
```

Prepare tighter or wider crops around the detected face:

```bash
morphace prep \
  images/raw \
  images/aligned \
  --x-scale 1.2 \
  --y-scale 1.3
```

After preparing your images, choose two aligned PNG files and pass them to
`morphace morph`:

```bash
morphace morph \
  images/aligned/person1_face01.png \
  images/aligned/person2_face01.png
```

#### Common options

- `--output-size` sets the square crop size in pixels. The default is `1024`.
- `--x-scale` controls how much horizontal context is included around the face.
- `--y-scale` controls how much vertical context is included around the face.
- `--em-scale` shifts the crop center from the eyes toward the mouth.
- `--alpha` uses transparent padding instead of reflected padding.
- `--force` overwrites existing prepared images.
- `--landmark-model` uses a [custom model path](https://morphace.daspyro.de/faq/alt-model-location/) instead of the [default download location](https://morphace.daspyro.de/faq/model-file-location/).

For full details, refer to
[prep CLI Reference](https://morphace.daspyro.de/cli-reference/prep/) on the
Morphace documentation site.

### morph

Use [`morphace morph`](https://morphace.daspyro.de/cli-reference/morph/) to generate an MP4 video that smoothly transitions from
one face image to another.

```bash
morphace morph images/first.png images/second.png
```

By default, Morphace writes the result to `morph.mp4`, creates a 5-second
video, and renders it at 30 frames per second.

The frame count is based on the selected duration and frame rate. With
the defaults, `--duration 5` and `--fps 30`, Morphace generates 150
frames. The first frame matches the first image, the last frame matches
the second image, and the frames between them move through evenly spaced
steps.

For best results, use images where the faces are similarly framed. If your
source images have different crops, angles, or face sizes, run them through
`morphace prep` first.

#### Examples

Create a morph video with the default settings:

```bash
morphace morph \
  images/first.png \
  images/second.png
```

Choose the output file:

```bash
morphace morph \
  images/first.png \
  images/second.png \
  --output output/friends_morph.mp4
```

Create a longer, smoother morph:

```bash
morphace morph \
  images/first.png \
  images/second.png \
  --duration 8 \
  --fps 60
```

Show the triangle mesh used for the face warp:

```bash
morphace morph \
  images/first.png \
  images/second.png \
  --show-mesh
```

#### Common options

- `--output` sets the MP4 output path. The default is `morph.mp4`.
- `--duration` sets the video length in seconds. The default is `5`.
- `--fps` sets the output frame rate. The default is `30`.
- `--show-mesh` draws the [face-warping mesh](https://morphace.daspyro.de/gallery/show-mesh/) over the generated video.
- `--force` overwrites an existing output video.
- `--landmark-model` uses a custom model path instead of the default download location.
- `--show-ffmpeg-output` shows FFmpeg encoder output instead of only errors.

For full details, refer to
[morph CLI Reference](https://morphace.daspyro.de/cli-reference/morph/) on the
Morphace documentation site.

## Tips for Best Results

- Use front-facing portraits when possible.
- Choose images with similar lighting and head angle.
- Run [`morphace prep`](https://morphace.daspyro.de/cli-reference/prep/) first if the faces are different sizes or crops.
- Use [`--show-mesh`](https://morphace.daspyro.de/gallery/show-mesh/) to inspect how the face geometry is being warped.

## License

The code in this project is licensed under the MIT license. See
[LICENSE.txt](https://github.com/pierow2k/morphace/blob/main/LICENSE.txt) for details.
