"""Manages the ASR model — Whisper ensemble (large-v3 + large-v2)."""

import io
import logging

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model settings ────────────────────────────────────────────────────────────
MODEL_DIR = "/models"          # pre-downloaded at Docker build time
PRIMARY_MODEL = "large-v3"    # highest accuracy; used as the main path
SECONDARY_MODEL = "large-v2"  # different training; catches different errors
DEVICE = "cuda"
COMPUTE_TYPE = "float16"       # ~3.1 GB VRAM each → ~6.2 GB total on T4

# ── Decoding settings ─────────────────────────────────────────────────────────
BEAM_SIZE = 5                  # balance accuracy vs speed
LANGUAGE = "en"

# Threshold below which the secondary model is also consulted.
# avg_logprob is in (-inf, 0]; -0.5 corresponds to roughly "uncertain".
LOW_CONF_THRESHOLD = -0.5

# If both models are below this, try a temperature-sampled fallback.
VERY_LOW_CONF_THRESHOLD = -1.0

# VAD keeps noisy silences from confusing the decoder.
VAD_PARAMS = {
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 400,
    "threshold": 0.35,         # lower → more aggressive noise suppression
}


class ASRManager:

    def __init__(self):
        logger.info("Loading primary model  : %s", PRIMARY_MODEL)
        self.primary = WhisperModel(
            PRIMARY_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_DIR,
            num_workers=2,
        )

        logger.info("Loading secondary model: %s", SECONDARY_MODEL)
        self.secondary = WhisperModel(
            SECONDARY_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_DIR,
            num_workers=2,
        )
        logger.info("✓ Both models loaded and ready.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes → normalised float32 mono numpy array."""
        buf = io.BytesIO(audio_bytes)
        audio, _ = sf.read(buf, dtype="float32")

        # Stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Peak-normalise to 0.95 so loud clips don't saturate the encoder.
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95

        return audio

    def _run_model(
        self,
        model: WhisperModel,
        audio: np.ndarray,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """
        Transcribe audio with one model.

        Returns
        -------
        text : str
            Full transcript (empty string if no speech detected).
        avg_logprob : float
            Mean per-token log-probability across all segments.
            Range: (-inf, 0]. Higher (closer to 0) = more confident.
        """
        segments, _ = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=VAD_PARAMS,
            word_timestamps=False,
            condition_on_previous_text=True,
            # Reject segments that are likely silence / hallucinations.
            no_speech_threshold=0.6,
            log_prob_threshold=-1.5,
            compression_ratio_threshold=2.4,
        )

        # Consume the generator — critical, results are lazy.
        seg_list = list(segments)

        if not seg_list:
            return "", -10.0

        text = " ".join(s.text.strip() for s in seg_list).strip()
        avg_logprob = float(np.mean([s.avg_logprob for s in seg_list]))
        return text, avg_logprob

    # ── Public API ─────────────────────────────────────────────────────────────

    def asr(self, audio_bytes: bytes) -> str:
        """
        Ensemble ASR pipeline:

        1. Run large-v3 (primary).
        2. If primary confidence is low, run large-v2 (secondary) and pick
           whichever is more confident.
        3. If both are very uncertain, retry primary with temperature=0.2
           (stochastic sampling often recovers noisy/OOV audio).

        This keeps latency low for clear audio while spending extra compute
        only on difficult samples.
        """
        audio = self._load_audio(audio_bytes)

        # ── Step 1: primary model ──────────────────────────────────────────
        primary_text, primary_conf = self._run_model(self.primary, audio)
        logger.debug("Primary  conf=%.3f  text=%r", primary_conf, primary_text[:60])

        # Fast path: primary is confident — return immediately.
        if primary_conf >= LOW_CONF_THRESHOLD:
            return primary_text

        # ── Step 2: secondary model (ensemble) ────────────────────────────
        secondary_text, secondary_conf = self._run_model(self.secondary, audio)
        logger.debug("Secondary conf=%.3f  text=%r", secondary_conf, secondary_text[:60])

        best_text = primary_text
        best_conf = primary_conf

        if secondary_conf > primary_conf:
            best_text = secondary_text
            best_conf = secondary_conf
            logger.debug("Secondary model preferred (%.3f > %.3f)", secondary_conf, primary_conf)

        # ── Step 3: temperature-sampled fallback ──────────────────────────
        # Both models are uncertain — attempt stochastic decoding on primary.
        if best_conf < VERY_LOW_CONF_THRESHOLD:
            fallback_text, fallback_conf = self._run_model(
                self.primary, audio, temperature=0.2
            )
            logger.debug("Fallback conf=%.3f  text=%r", fallback_conf, fallback_text[:60])
            if fallback_conf > best_conf:
                logger.debug("Fallback preferred (%.3f > %.3f)", fallback_conf, best_conf)
                return fallback_text

        return best_text