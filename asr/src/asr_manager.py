"""Manages the ASR model — faster-whisper ensemble + XGBoost."""

import io
import logging
import os

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INITIAL_PROMPT = "Military communication. Callsigns, grid references, SITREP, Roger, Wilco, Oscar Mike, Lima Charlie."
MODEL_DIR  = "/app/models"
PRIMARY    = "large-v3"
SECONDARY  = "large-v2"
DEVICE     = "cuda"
COMPUTE     = "int8"
BEAM_SIZE  = 4
LANGUAGE   = "en"
XGB_PATH   = "/app/xgb_selector.joblib"
XGB_THRESH = 0.42

VAD_PARAMS = {
    "min_silence_duration_ms": 200,
    "speech_pad_ms":           200,
    "threshold":               0.55,
}


class CalibratedModel:
    """Included here so joblib can unpickle the XGBoost model."""
    def __init__(self, model, iso):
        self._model = model
        self._iso   = iso

    def predict_proba(self, X):
        raw = self._model.predict_proba(X)[:, 1]
        cal = self._iso.predict(raw)
        return np.column_stack([1 - cal, cal])


def _feats(segs):
    if not segs:
        return np.zeros(9, dtype=np.float32)
    lp  = [s.avg_logprob       for s in segs]
    ns  = [s.no_speech_prob    for s in segs]
    cr  = [s.compression_ratio for s in segs]
    txt = " ".join(s.text.strip() for s in segs)
    return np.array([
        np.mean(lp), np.min(lp), np.std(lp) if len(lp) > 1 else 0.0,
        np.mean(ns), np.max(ns),
        np.mean(cr), np.max(cr),
        float(len(segs)), float(len(txt.split())),
    ], dtype=np.float32)


class ASRManager:

    def __init__(self):
        logger.info("Loading %s from %s", PRIMARY, MODEL_DIR)
        self.primary = WhisperModel(
            PRIMARY, device=DEVICE, compute_type=COMPUTE,
            download_root=MODEL_DIR, num_workers=2,
        )
        logger.info("Loading %s", SECONDARY)
        self.secondary = WhisperModel(
            SECONDARY, device=DEVICE, compute_type=COMPUTE,
            download_root=MODEL_DIR, num_workers=2,
        )

        self.xgb     = None
        self.xgb_iso = None
        if os.path.exists(XGB_PATH):
            try:
                import joblib
                payload          = joblib.load(XGB_PATH)
                self.xgb         = payload["xgb_model"]
                self.xgb_iso     = payload["iso"]
                global XGB_THRESH
                XGB_THRESH       = payload.get("threshold", XGB_THRESH)
                logger.info("XGBoost loaded. Threshold=%.2f", XGB_THRESH)
            except Exception as e:
                logger.warning("XGBoost load failed: %s — using logprob fallback", e)

        logger.info("ASRManager ready.")

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        audio, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95
        return audio

    def _run(self, model, audio, temperature=0.0):
        segs, _ = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            initial_prompt=INITIAL_PROMPT,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=VAD_PARAMS,
            word_timestamps=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-0.6,
            compression_ratio_threshold=1.6,
        )
        segs = list(segs)
        if not segs:
            return "", -10.0, _feats([])
        text = " ".join(s.text.strip() for s in segs).strip()
        conf = float(np.mean([s.avg_logprob for s in segs]))
        return text, conf, _feats(segs)

    def asr(self, audio_bytes: bytes) -> str:
        audio = self._load_audio(audio_bytes)

        p_text, p_conf, p_feat = self._run(self.primary,   audio)
        s_text, s_conf, s_feat = self._run(self.secondary, audio)

        if self.xgb is not None:
            delta    = p_feat - s_feat
            combined = np.concatenate([p_feat, s_feat, delta])[np.newaxis, :]
            raw_prob = float(self.xgb.predict_proba(combined)[0][1])
            prob     = float(self.xgb_iso.predict([raw_prob])[0]) if self.xgb_iso else raw_prob
            use_primary = prob >= 0.5
            confident   = max(prob, 1 - prob) >= XGB_THRESH
            if confident:
                best_text = p_text if use_primary else s_text
                best_conf = p_conf if use_primary else s_conf
            else:
                best_text = p_text if p_conf >= s_conf else s_text
                best_conf = max(p_conf, s_conf)
        else:
            best_text = p_text if p_conf >= s_conf else s_text
            best_conf = max(p_conf, s_conf)

        # Temperature fallback for very uncertain outputs
        if best_conf < -1.0:
            fb_text, fb_conf, _ = self._run(self.primary, audio, temperature=0.2)
            if fb_conf > best_conf:
                return fb_text

        return best_text
