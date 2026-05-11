"""Manages the ASR model — faster-whisper ensemble + XGBoost meta-selector."""

import io
import logging
import os

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model settings ────────────────────────────────────────────────────────────
MODEL_DIR        = "/models"
XGB_MODEL_PATH   = "/app/xgb_selector.joblib"   # written by train_xgb.py
PRIMARY_MODEL    = "large-v3"
SECONDARY_MODEL  = "large-v2"
DEVICE           = "cuda"
COMPUTE_TYPE     = "float16"    # ~3.1 GB VRAM each → ~6.2 GB total on T4

# ── Decoding settings ─────────────────────────────────────────────────────────
BEAM_SIZE = 5
LANGUAGE  = "en"

# If XGBoost's win-probability for the chosen model is below this, fall back
# to avg_logprob comparison (XGBoost is uncertain — trust raw confidence).
XGB_CONFIDENCE_THRESHOLD = 0.60

# Temperature fallback: used when both models are very uncertain.
VERY_LOW_CONF_THRESHOLD = -1.0

VAD_PARAMS = {
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 400,
    "threshold": 0.35,
}


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(seg_list: list) -> np.ndarray:
    """
    Turn a list of faster-whisper Segment objects into a 1-D feature vector.

    Features (9 total):
        0  avg_logprob_mean   — mean per-token log-prob across segments
        1  avg_logprob_min    — worst-segment confidence
        2  avg_logprob_std    — spread of confidence
        3  no_speech_mean     — mean probability that segments are silence
        4  no_speech_max      — worst silence-probability segment
        5  compression_mean   — mean text compression ratio
        6  compression_max    — highest compression (possible hallucination)
        7  num_segments       — how many speech segments were found
        8  num_words          — total word count in transcript
    """
    if not seg_list:
        return np.array([-10., -10., 0., 1., 1., 0., 0., 0., 0.], dtype=np.float32)

    logprobs     = [s.avg_logprob       for s in seg_list]
    no_speech    = [s.no_speech_prob    for s in seg_list]
    compression  = [s.compression_ratio for s in seg_list]
    text         = " ".join(s.text.strip() for s in seg_list)

    return np.array([
        np.mean(logprobs),
        np.min(logprobs),
        np.std(logprobs) if len(logprobs) > 1 else 0.0,
        np.mean(no_speech),
        np.max(no_speech),
        np.mean(compression),
        np.max(compression),
        float(len(seg_list)),
        float(len(text.split())),
    ], dtype=np.float32)


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

        # ── XGBoost meta-selector (optional) ──────────────────────────────
        self.xgb = None
        if os.path.exists(XGB_MODEL_PATH):
            try:
                import joblib
                self.xgb = joblib.load(XGB_MODEL_PATH)
                logger.info("✓ XGBoost meta-selector loaded from %s", XGB_MODEL_PATH)
            except Exception as e:
                logger.warning("Could not load XGBoost model: %s — using fallback", e)
        else:
            logger.info(
                "XGBoost model not found at %s — using avg_logprob fallback. "
                "Run train_xgb.py inside the container to enable it.",
                XGB_MODEL_PATH,
            )

        logger.info("✓ ASRManager ready.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """WAV bytes → normalised float32 mono numpy array."""
        buf = io.BytesIO(audio_bytes)
        audio, _ = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95
        return audio

    def _run_model(
        self,
        model: WhisperModel,
        audio: np.ndarray,
        temperature: float = 0.0,
    ) -> tuple[str, float, np.ndarray]:
        """
        Transcribe audio with one model.

        Returns
        -------
        text         : str        — full transcript
        avg_logprob  : float      — mean per-token log-probability
        features     : np.ndarray — 9-dim feature vector for XGBoost
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
            no_speech_threshold=0.6,
            log_prob_threshold=-1.5,
            compression_ratio_threshold=2.4,
        )

        seg_list = list(segments)   # consume lazy generator

        if not seg_list:
            return "", -10.0, _extract_features([])

        text        = " ".join(s.text.strip() for s in seg_list).strip()
        avg_logprob = float(np.mean([s.avg_logprob for s in seg_list]))
        features    = _extract_features(seg_list)
        return text, avg_logprob, features

    def _xgb_select(
        self,
        feat_primary: np.ndarray,
        feat_secondary: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Ask XGBoost which model to trust.

        The model was trained with label=1 when primary (large-v3) had lower
        WER, label=0 when secondary (large-v2) was better.

        Returns
        -------
        use_primary  : bool   — True → use primary output
        probability  : float  — confidence of the decision (0.5 = coin-flip)
        """
        # Feature vector: [primary_features | secondary_features | delta_features]
        delta    = feat_primary - feat_secondary
        combined = np.concatenate([feat_primary, feat_secondary, delta])[np.newaxis, :]

        prob_primary = float(self.xgb.predict_proba(combined)[0][1])
        use_primary  = prob_primary >= 0.5
        confidence   = prob_primary if use_primary else (1.0 - prob_primary)
        return use_primary, confidence

    # ── Public API ─────────────────────────────────────────────────────────────

    def asr(self, audio_bytes: bytes) -> str:
        """
        Ensemble ASR pipeline:

        1. Always run both large-v3 and large-v2.
        2. If XGBoost selector is available AND confident (≥ threshold):
               use its choice.
        3. Otherwise fall back to avg_logprob comparison.
        4. If the chosen output has very low confidence, retry primary with
           temperature=0.2 (stochastic sampling recovers noisy/OOV audio).
        """
        audio = self._load_audio(audio_bytes)

        # ── Step 1: run both models ────────────────────────────────────────
        p_text, p_conf, p_feat = self._run_model(self.primary,   audio)
        s_text, s_conf, s_feat = self._run_model(self.secondary, audio)

        logger.debug("Primary   conf=%.3f  text=%r", p_conf, p_text[:60])
        logger.debug("Secondary conf=%.3f  text=%r", s_conf, s_text[:60])

        # ── Step 2: select transcript ──────────────────────────────────────
        if self.xgb is not None:
            use_primary, xgb_prob = self._xgb_select(p_feat, s_feat)

            if xgb_prob >= XGB_CONFIDENCE_THRESHOLD:
                # XGBoost is confident — trust it.
                best_text = p_text if use_primary else s_text
                best_conf = p_conf if use_primary else s_conf
                logger.debug(
                    "XGBoost selected %s (p=%.3f)",
                    "primary" if use_primary else "secondary",
                    xgb_prob,
                )
            else:
                # XGBoost is uncertain (near 50/50) — fall back to raw confidence.
                logger.debug("XGBoost uncertain (p=%.3f) — using avg_logprob", xgb_prob)
                if p_conf >= s_conf:
                    best_text, best_conf = p_text, p_conf
                else:
                    best_text, best_conf = s_text, s_conf
        else:
            # No XGBoost model yet — use avg_logprob comparison.
            if p_conf >= s_conf:
                best_text, best_conf = p_text, p_conf
            else:
                best_text, best_conf = s_text, s_conf

        # ── Step 3: temperature fallback for very noisy audio ─────────────
        if best_conf < VERY_LOW_CONF_THRESHOLD:
            fb_text, fb_conf, _ = self._run_model(self.primary, audio, temperature=0.2)
            logger.debug("Fallback conf=%.3f  text=%r", fb_conf, fb_text[:60])
            if fb_conf > best_conf:
                return fb_text

        return best_text