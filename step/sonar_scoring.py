import json
import tempfile
from typing import Any, Dict, List

import torch
from base import BaseStep
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from step_decorator import Step


@Step("sonar_scoring")
class SonarScoring(BaseStep):
    def initialise(
        self,
        infra: List[Dict],
        speech_lang_code: str,
        text_lang_code: str,
        batch_size: int,
    ) -> Any:
        self.infra = infra

        self.text_lang_code = text_lang_code
        self.batch_size = batch_size

        device = torch.device("cuda")
        # Load text model and tokenizer
        self.text_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

        # Load audio model
        self.audio_model = SpeechToEmbeddingModelPipeline(
            encoder=speech_lang_code,
            device=device,
        )

    def run(self) -> Any:
        tmp_dir: tempfile.TemporaryDirectory = self.get_state("manifest_tmp_dir")
        with open(tmp_dir.name + "/manifest.jsonl") as fhand:
            lines = [json.loads(line) for line in fhand.readlines()]

        paths = []
        texts = []

        for idx, line in enumerate(lines):
            if line["rnnt_pred_text"] == "":
                lines[idx]["sonar_score"] = 0.0
                continue

            paths.append(line["audio_filepath"])
            texts.append(line["rnnt_pred_text"])

        if len(paths) > 0:
            try:
                text_embeddings = self.text_model.predict(
                    texts,
                    source_lang=self.text_lang_code,
                    batch_size=self.batch_size,
                    progress_bar=True,
                )
            except Exception as e:
                if hasattr(e, "__cause__"):
                    print(f"Nested exception: {e.__cause__}")
                raise e

            audio_embeddings = self.audio_model.predict(
                paths,
                batch_size=self.batch_size,
                n_prefetched_batches=self.batch_size,
                progress_bar=True,
            )

            sonar_scores = (
                torch.nn.functional.cosine_similarity(
                    text_embeddings, audio_embeddings, dim=1
                )
                .cpu()
                .tolist()
            )

        sonar_idx = 0
        for line_idx in range(len(lines)):
            if lines[line_idx].get("sonar_score") is None:
                lines[line_idx]["sonar_score"] = sonar_scores[sonar_idx]
                sonar_idx += 1

            lines[line_idx] = json.dumps(lines[line_idx])

        with open(tmp_dir.name + "/manifest.jsonl", "w") as fhand:
            fhand.write("\n".join(lines))
