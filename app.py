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

    from deink import Inpainter, TattooSegmenter, remove_tattoo

    @functools.lru_cache(maxsize=1)
    def get_segmenter():
        return TattooSegmenter()

    @functools.lru_cache(maxsize=1)
    def get_inpainter():
        return Inpainter()

    def to_png(img: "Image.Image") -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    return (
        Image,
        get_inpainter,
        get_segmenter,
        io,
        mo,
        remove_tattoo,
        time,
        to_png,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🖋️ deinked — tattoo remover
        Upload a photo. The tool locates the tattoo (GroundingDINO + SAM) and paints it
        out (LaMa or SDXL). If auto-detect misses, upload a black/white mask instead
        (white = area to remove).
        """
    )
    return


@app.cell
def _(mo):
    upload = mo.ui.file(kind="button", label="Upload image", filetypes=[".png", ".jpg", ".jpeg"])
    mask_upload = mo.ui.file(
        kind="button", label="Optional mask (white = remove)", filetypes=[".png", ".jpg", ".jpeg"]
    )
    backend = mo.ui.dropdown(["lama", "sdxl"], value="lama", label="Inpaint backend")
    prompt = mo.ui.text(value="a tattoo.", label="Detection prompt")
    dilate = mo.ui.slider(0, 40, value=15, label="Mask grow (px)")
    feather = mo.ui.slider(0, 25, value=5, label="Feather (px)")
    run = mo.ui.run_button(label="Remove tattoo")

    controls = mo.vstack(
        [
            mo.hstack([upload, mask_upload], justify="start"),
            mo.hstack([backend, prompt], justify="start"),
            mo.hstack([dilate, feather], justify="start"),
            run,
        ]
    )
    controls
    return backend, dilate, feather, mask_upload, prompt, run, upload


@app.cell
def _(
    Image,
    backend,
    dilate,
    feather,
    get_inpainter,
    get_segmenter,
    io,
    mask_upload,
    mo,
    prompt,
    run,
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
        dilate=dilate.value,
        feather=feather.value,
        segmenter=get_segmenter(),
        inpainter=get_inpainter(),
    )
    elapsed = time.time() - t0

    if not result.found:
        status = mo.md(f"⚠️ No tattoo found (backend `{backend.value}`, {elapsed:.1f}s). "
                       "Try lowering the prompt threshold or upload a mask.")
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
