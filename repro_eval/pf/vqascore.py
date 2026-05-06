"""VQAScore — Lin et al., ECCV 2024 (arXiv:2404.01291).

Recipe: P("Yes" | image, "Does this figure show '{prompt}'? Please answer
yes or no.") computed by a generative VLM (default `clip-flant5-xxl`,
~3B params). Single scalar in [0, 1] per pair.

Install: `pip install t2v-metrics` (https://github.com/linzhiqiu/t2v_metrics).
Cache dir defaults to ~/.cache/torch/hub/checkpoints unless overridden via
HF_HOME / TORCH_HOME.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

DEFAULT_MODEL = "clip-flant5-xxl"


@dataclass
class VQAScoreScorer:
    model: object  # t2v_metrics.VQAScore instance
    model_id: str

    def score(self, image: Image.Image, prompt_text: str) -> dict:
        # The package's VQAScore accepts in-memory PIL images via the
        # `images=` kwarg in recent versions; older versions wanted file
        # paths. We pass the PIL image directly; if that fails, write a
        # temp PNG and pass the path.
        try:
            scores = self.model(images=[image], texts=[prompt_text])
        except Exception:
            import io, tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                image.save(tf, format="PNG")
                path = tf.name
            try:
                scores = self.model(images=[path], texts=[prompt_text])
            finally:
                os.unlink(path)
        # `scores` shape: (n_images, n_texts) tensor in [0, 1].
        s = float(scores.flatten()[0])
        return {
            "score": s,
            "extras": {
                "score_type": "vqascore_yes_prob",
                "model_id": self.model_id,
            },
        }


def load_scorer(
    model_id: str = DEFAULT_MODEL,
    device: str = "cuda",
) -> VQAScoreScorer:
    # t2v_metrics' __init__ does a hard `shutil.which("ffmpeg")` check at
    # import time. If the system PATH lacks ffmpeg (no sudo / no conda),
    # fall back to the static binary bundled by imageio-ffmpeg.
    import os
    import shutil
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            ffbin = imageio_ffmpeg.get_ffmpeg_exe()
            os.environ["PATH"] = os.path.dirname(ffbin) + os.pathsep + os.environ.get("PATH", "")
            # imageio-ffmpeg names the binary `ffmpeg-linux-x86_64-v7.0.2`,
            # not `ffmpeg`. shutil.which looks for the exact name, so
            # symlink the expected name into a writable dir on PATH.
            if shutil.which("ffmpeg") is None:
                import tempfile
                shim_dir = os.path.join(tempfile.gettempdir(), "masc_repro_ffmpeg_shim")
                os.makedirs(shim_dir, exist_ok=True)
                shim = os.path.join(shim_dir, "ffmpeg")
                if not os.path.exists(shim):
                    os.symlink(ffbin, shim)
                os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
        except ImportError:
            pass
    import t2v_metrics
    model = t2v_metrics.VQAScore(model=model_id, device=device)
    return VQAScoreScorer(model=model, model_id=model_id)
