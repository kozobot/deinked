"""deinked — interactive tattoo remover (marimo app).

Run with:
    conda activate deinked
    marimo run app.py        # app mode
    marimo edit app.py       # editable notebook

Upload an image, let it auto-detect the tattoo (or upload your own mask), and remove it.
Heavy models are loaded once and cached across reruns.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import functools
    import io
    import time

    import marimo as mo
    from PIL import Image

    from deink import Inpainter, TattooMaskSegmenter, TattooSegmenter, remove_tattoo

    @functools.lru_cache(maxsize=1)
    def get_segmenter():
        return TattooSegmenter()

    @functools.lru_cache(maxsize=1)
    def get_mask_segmenter():
        return TattooMaskSegmenter()

    @functools.lru_cache(maxsize=1)
    def get_inpainter():
        return Inpainter()

    # Offer the custom-seg localization paths only when a fine-tuned checkpoint exists, and
    # default to "seg" when it does — it beats the box+SAM baseline on recall *and* precision.
    seg_available = TattooMaskSegmenter.available()
    localizer_choices = ["seg", "box", "seg+box"] if seg_available else ["box"]
    localizer_default = "seg" if seg_available else "box"

    def to_png(img: "Image.Image") -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    return (
        Image,
        get_inpainter,
        get_mask_segmenter,
        get_segmenter,
        io,
        localizer_choices,
        localizer_default,
        mo,
        remove_tattoo,
        seg_available,
        time,
        to_png,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🖋️ deinked — tattoo remover
        Upload a photo. The tool locates the tattoo (GroundingDINO + SAM) and paints it
        out (LaMa, SDXL, `twostage` — LaMa structure then a light SDXL texture pass — or
        `auto`, which routes each region by size: small blobs to LaMa, large sleeves to
        two-stage). If auto-detect misses, upload a black/white mask instead
        (white = area to remove).
        """
    )
    return


@app.cell
def _(localizer_choices, localizer_default, mo, seg_available):
    upload = mo.ui.file(kind="button", label="Upload image", filetypes=[".png", ".jpg", ".jpeg"])
    mask_upload = mo.ui.file(
        kind="button", label="Optional mask (white = remove)", filetypes=[".png", ".jpg", ".jpeg"]
    )
    # Default to "auto": crop-to-region validation showed LaMa is cleanest/fastest on small
    # plain-skin tattoos while large ones need structure reconstruction, so per-component routing
    # (small -> LaMa, large -> two-stage LaMa+SDXL) gives the best visual result across image
    # types. See the crop-to-region findings. "twostage" forces the LaMa-structure -> SDXL-texture
    # fill on the whole mask.
    backend = mo.ui.dropdown(
        ["lama", "sdxl", "flux", "auto", "twostage"], value="auto", label="Inpaint backend"
    )
    localizer = mo.ui.dropdown(
        localizer_choices,
        value=localizer_default,
        label="Localizer" + ("" if seg_available else " (train seg model to enable seg)"),
    )
    detector = mo.ui.dropdown(["gdino", "owlv2", "ensemble"], value="gdino", label="Detector")
    prompt = mo.ui.text(value="a tattoo.", label="Detection prompt")
    tile = mo.ui.checkbox(value=False, label="Tile detect (slower, better recall)")
    dilate = mo.ui.slider(0, 40, value=8, label="Mask grow (px)")
    feather = mo.ui.slider(0, 25, value=5, label="Feather (px)")
    box_threshold = mo.ui.slider(0.0, 0.6, step=0.05, value=0.25, label="Box threshold")
    text_threshold = mo.ui.slider(0.0, 0.6, step=0.05, value=0.2, label="Text threshold")
    max_area_frac = mo.ui.slider(0.05, 1.0, step=0.05, value=0.25, label="Max box area frac")
    seg_threshold = mo.ui.slider(0.3, 0.95, step=0.05, value=0.5,
                                 label="Seg threshold (higher = tighter mask)")
    run = mo.ui.run_button(label="Remove tattoo")

    controls = mo.vstack(
        [
            mo.hstack([upload, mask_upload], justify="start"),
            mo.hstack([backend, localizer, detector, prompt, tile], justify="start"),
            mo.hstack([dilate, feather], justify="start"),
            mo.accordion(
                {
                    "▸ Advanced detection (lower thresholds = more recall, more false positives)":
                        mo.hstack([box_threshold, text_threshold, max_area_frac, seg_threshold],
                                  justify="start")
                }
            ),
            run,
        ]
    )
    controls
    return (
        backend,
        box_threshold,
        detector,
        dilate,
        feather,
        localizer,
        mask_upload,
        max_area_frac,
        prompt,
        run,
        seg_threshold,
        text_threshold,
        tile,
        upload,
    )


@app.cell
def _(
    Image,
    backend,
    box_threshold,
    detector,
    dilate,
    feather,
    get_inpainter,
    get_mask_segmenter,
    get_segmenter,
    io,
    localizer,
    mask_upload,
    max_area_frac,
    mo,
    prompt,
    run,
    seg_threshold,
    text_threshold,
    tile,
    time,
    to_png,
    remove_tattoo,
    upload,
):
    mo.stop(not run.value, mo.md("*Upload an image and press **Remove tattoo**.*"))
    mo.stop(not upload.value, mo.md("⚠️ No image uploaded yet."))

    src_bytes = upload.value[0].contents
    image = Image.open(io.BytesIO(src_bytes)).convert("RGB")

    user_mask = None
    if mask_upload.value:
        user_mask = Image.open(io.BytesIO(mask_upload.value[0].contents)).convert("L")

    t0 = time.time()
    result = remove_tattoo(
        image,
        backend=backend.value,
        prompt=prompt.value,
        mask=user_mask,
        localizer=localizer.value,
        seg_threshold=seg_threshold.value,
        detector=detector.value,
        tile=tile.value,
        dilate=dilate.value,
        feather=feather.value,
        box_threshold=box_threshold.value,
        text_threshold=text_threshold.value,
        max_area_frac=max_area_frac.value,
        segmenter=get_segmenter(),
        mask_segmenter=get_mask_segmenter(),
        inpainter=get_inpainter(),
    )
    elapsed = time.time() - t0

    if not result.found:
        status = mo.md(f"⚠️ No tattoo found (backend `{backend.value}`, {elapsed:.1f}s). "
                       "Lower the box/text thresholds under **Advanced detection**, try a "
                       "compound prompt, or upload a mask.")
    else:
        status = mo.md(f"✅ Done in {elapsed:.1f}s using `{backend.value}`.")

    view = mo.hstack(
        [
            mo.vstack([mo.md("**Before**"), mo.image(to_png(image), width=380)]),
            mo.vstack([mo.md("**After**"), mo.image(to_png(result.image), width=380)]),
            mo.vstack([mo.md("**Mask**"), mo.image(to_png(result.mask.convert("RGB")), width=200)]),
        ],
        justify="start",
    )
    mo.vstack([status, view])
    return


if __name__ == "__main__":
    app.run()
