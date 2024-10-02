import json
import os
from typing import Dict

import enchant
import glob2
from base_classes import BaseStep
from joblib import Parallel, delayed
from pydub import AudioSegment
from step_decorator import step
from tqdm import tqdm


@step("build_final_manifest")
class BuildFinalManifest(BaseStep):
    def initialise(
        self,
        infra: Dict[str, int],
        audio_out_folder: str,
        manifest_out_folder: str,
        segment_path: str,
        audio_path: str,
        language: str,
        subdiv: str,
    ):
        self.infra = infra
        self.audio_out_folder = audio_out_folder
        self.manifest_out_folder = manifest_out_folder
        self.segment_path = segment_path
        self.audio_path = audio_path
        self.language = language
        self.subdiv = subdiv
        os.makedirs(self.audio_out_folder, exist_ok=True)
        os.makedirs(self.manifest_out_folder, exist_ok=True)

    def _read_ctm_file(self, ctm_path: str):
        with open(ctm_path, "r") as fhand:
            lines = fhand.readlines()

        data = [line.split(" ")[2:] for line in lines]
        return data

    def _calculate_alignment_score(self, text: str, pred_text: str):
        return 1 - (
            enchant.utils.levenshtein(text, pred_text) / (len(text) + len(pred_text))
        )

    def _chunk_audio(self, ctm_path: str):
        data = self._read_ctm_file(ctm_path)

        audio_name = ctm_path.split("/")[-1][:-4]
        audio_path = self.audio_path + "/" + audio_name + ".wav"

        os.makedirs(f"{self.audio_out_folder}/{audio_name}", exist_ok=True)
        manifest = []

        sound: AudioSegment = AudioSegment.from_wav(audio_path)
        for idx, data in enumerate(data):
            start = round(float(data[0]) * 1000)
            end = round(float(data[1]) * 1000) + start
            sound_clip = sound[start:end]
            sound_clip.export(
                f"{self.audio_out_folder}/{audio_name}/{audio_name}_chunk_{str(idx)}.wav",
                "wav",
            )

            text = data[2].replace("<space>", " ")
            pred_text = data[-1].strip().replace("<space>", " ")
            alignment_score = self._calculate_alignment_score(text, pred_text)

            manifest.append(
                {
                    "audio_filepath": f"{self.audio_out_folder}/{audio_name}/{audio_name}_chunk_{str(idx)}.wav",
                    "text": text,
                    "pred_text": pred_text,
                    "duration": float(data[1]),
                    "alignment_score": alignment_score,
                }
            )

        return manifest

    def run(self):
        manifest_data = Parallel(n_jobs=self.infra["cpu"])(
            delayed(self._chunk_audio)(ctm_path)
            for ctm_path in tqdm(glob2.iglob(self.segment_path + "/*.ctm"))
        )

        full_manifest = []
        for manifest in manifest_data:
            full_manifest.extend(manifest)

        with open(
            f"{self.manifest_out_folder}/manifest_{self.language}_{self.subdiv}.jsonl",
            "w+",
        ) as fhand:
            for manifest in full_manifest:
                if not manifest:
                    continue

                fhand.write(json.dumps(manifest))
                fhand.write("\n")
