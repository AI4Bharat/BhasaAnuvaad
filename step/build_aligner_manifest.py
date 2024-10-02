import json
import os
import re
from typing import Any, Dict, List

import torch
from base import BaseStep
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from step_decorator import Step
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


@Step("build_aligner_manifest")
class BuildAlignerManifest(BaseStep):
    def initialise(
        self,
        infra: Dict[str, int],
        model_id: str,
        split_chars: List[str],
    ) -> Any:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )
        self.output_manifest_path = self.get_state("aligner_manifest_path")
        self.audio_out_path = self.get_state("final_audio_path")
        self.language = self.get_state("aligner_language")
        self.split_chars = split_chars

        os.makedirs(self.output_manifest_path, exist_ok=True)
        os.makedirs(f"{self.audio_out_path}/{self.language}", exist_ok=True)

    def _process_audio(self, in_path: str, out_path: str):
        sound: AudioSegment = AudioSegment.from_file(in_path.strip())
        sound = pydub_normalize(sound.set_frame_rate(16000).set_channels(1))

        sound.export(out_path, "wav")

        return sound.duration_seconds

    def _get_transcript(self, audio_path: str):
        result = self.model(audio_path)
        return result["text"]

    def _clean_text(self, text: str):
        text = re.sub("[-_\n\s]+", " ", text)
        sents = [
            sent.strip()
            for sent in re.split("[" + "".join(self.split_chars) + "]", text)
            if sent.strip() != ""
        ]
        return "ɘ".join(sents)

    def run(self) -> Any:
        out_manifest_path = (
            f"{self.output_manifest_path}/manifest_{self.language}.jsonl"
        )

        completed_lines = 0
        if os.path.exists(out_manifest_path):
            with open(out_manifest_path, "r") as fhand:
                completed_lines = len(fhand.readlines())

        with open(self.get_state("input_manifest_path"), "r") as fhand:
            lines = fhand.readlines()

        for line in tqdm(range(completed_lines, len(lines))):
            js = json.loads(lines[line])
            in_path = js["alignment_audio_path"].strip()
            wav_name = in_path.split("/")[-1][:-3] + "wav"
            wav_path = f"{self.audio_out_path}/{self.language}/{wav_name}"
            duration = self._process_audio(in_path, wav_path)

            if js.get("alignment_text_path") is None:
                text = self._get_transcript(wav_path)
            else:
                with open(js["alignment_text_path"]) as fhand:
                    text = fhand.read().strip()

            json_line = json.dumps(
                {
                    "audio_filepath": wav_path,
                    "text": self._clean_text(text),
                    "duration": duration,
                }
            )

            with open(out_manifest_path, "a+") as fhand:
                fhand.write(json_line)
