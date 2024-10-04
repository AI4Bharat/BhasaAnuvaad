import json
import os
import re
from typing import Any, Dict, List, Tuple

import enchant
import faiss
import numpy as np
import torch
from base import BaseStep
from pydub import AudioSegment
from step_decorator import Step
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder


@Step("build_final_manifest")
class BuildFinalManifest(BaseStep):
    def initialise(
        self,
        infra: Dict[str, int],
        batch_size: int,
        knn_neighborhood: int,
        margin_algorithm: str,
        sonar_device: str,
        mining_device: str,
    ):
        self.infra = infra
        self.knn_neighborhood = knn_neighborhood
        self.margin_algorithm = margin_algorithm
        self.batch_size = batch_size
        self.sonar_device = sonar_device
        self.mining_device = mining_device

        self.input_manifest_path = self.get_state("input_manifest_path")
        self.manifest_out_folder = self.get_state("final_manifest")
        # Load text model and tokenizer
        self.encoder = M2M100Encoder.from_pretrained(
            "cointegrated/SONAR_200_text_encoder"
        ).to(self.sonar_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/SONAR_200_text_encoder"
        )
        self.sonar_lang_mapping = {"en": "eng_Latn", "hi": "hin_Deva"}

        self.language = self.get_state("aligner_language")
        self.aligner_language_id = self.get_state("aligner_language_id")
        self.segment_manifest_path = (
            self.get_state("aligned_output_path")
            + "/"
            + self.language
            + f"/manifest_{self.language}_with_output_file_paths.json"
        )

        os.makedirs(self.manifest_out_folder, exist_ok=True)

    def _read_ctm_file(self, ctm_path: str):
        with open(ctm_path, "r") as fhand:
            lines = fhand.readlines()

        data = []
        for line in lines:
            parts = line.split(" ")

            # 2 - start timestamp sec, 3 - duration sec, 4 - aligned text, 8 - aligned pred text
            data.append(
                (
                    parts[2],
                    parts[3],
                    parts[4].replace("<space>", " ").strip(),
                    parts[8].replace("<space>", " ").strip(),
                )
            )

        return data

    def _read_text(self, text_path: str, separators: List[str]):
        with open(text_path, "r") as fhand:
            text = fhand.read()
        text = re.sub("[-_\n\s]+", " ", text)
        sents = [
            sent.strip()
            for sent in re.split("[" + "".join(separators) + "]", text)
            if sent.strip() != ""
        ]

        return sents

    def _knnCPU(self, x, y, k):
        dim = x.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(y)
        sim, ind = idx.search(x, k)
        return sim, ind

    def _knnGPU(self, x, y, k, mem=5 * 1024 * 1024 * 1024):
        dim = x.shape[1]
        batch_size = mem // (dim * 4)
        sim = np.zeros((x.shape[0], k), dtype=np.float32)
        ind = np.zeros((x.shape[0], k), dtype=np.int64)
        for xfrom in range(0, x.shape[0], batch_size):
            xto = min(xfrom + batch_size, x.shape[0])
            bsims, binds = [], []
            for yfrom in range(0, y.shape[0], batch_size):
                yto = min(yfrom + batch_size, y.shape[0])
                idx = faiss.IndexFlatIP(dim)
                idx = faiss.index_cpu_to_all_gpus(idx)
                idx.add(y[yfrom:yto])
                bsim, bind = idx.search(x[xfrom:xto], min(k, yto - yfrom))
                bsims.append(bsim)
                binds.append(bind + yfrom)
                del idx
            bsims = np.concatenate(bsims, axis=1)
            binds = np.concatenate(binds, axis=1)
            aux = np.argsort(-bsims, axis=1)
            for i in range(xfrom, xto):
                for j in range(k):
                    sim[i, j] = bsims[i - xfrom, aux[i - xfrom, j]]
                    ind[i, j] = binds[i - xfrom, aux[i - xfrom, j]]
        return sim, ind

    def _score(self, x, y, fwd_mean, bwd_mean, margin):
        return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)

    def _score_candidates(self, x, y, candidate_inds, fwd_mean, bwd_mean, margin):
        scores = np.zeros(candidate_inds.shape)
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                k = candidate_inds[i, j]
                scores[i, j] = self._score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)

        return scores

    def _get_pairs(self, fwd_scores: np.ndarray, x2y_ind: np.ndarray):
        seen_src = set()
        skipped = set()
        pairs = list()

        # first match best
        for i in range(len(fwd_scores)):
            max_ind = np.argmax(fwd_scores[i])

            # if the index was already chosen, skip it for now
            if x2y_ind[i, max_ind] in seen_src:
                skipped.add(i)
                continue

            seen_src.add(x2y_ind[i, max_ind])
            pairs.append((i, x2y_ind[i, max_ind], fwd_scores[i, max_ind]))

        # then match second best
        for i in skipped:
            done = False
            fwd_ind = np.argsort(fwd_scores[i])

            # loop across all the k - 1 indices and choose the one
            # that has the highest score and hasn't been already chosen
            for j in range(len(fwd_ind) - 2, -1, -1):
                max_ind = fwd_ind[j]
                if x2y_ind[i, max_ind] in seen_src:
                    continue

                done = True
                pairs.append((i, x2y_ind[i, max_ind], fwd_scores[i, max_ind]))
                seen_src.add(x2y_ind[i, max_ind])
                break

            # if all of the remaining indices were chosen, then we have
            # exhausted all possible neighbours. Set index to -1
            if not done:
                pairs.append((i, -1, -1))

        pairs = sorted(pairs, key=lambda x: x[0], reverse=False)

        return pairs

    def _calculate_alignment_score(self, text: str, pred_text: str):
        return 1 - (
            enchant.utils.levenshtein(text, pred_text) / (len(text) + len(pred_text))
        )

    def _encode_mean_pool(self, texts: List[str], lang_id: str, norm=False):
        lang = self.sonar_lang_mapping[lang_id]
        self.tokenizer.src_lang = lang

        embs = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(self.sonar_device)
                seq_embs = self.encoder(**batch).last_hidden_state
                mask = batch.attention_mask
                mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(
                    -1
                ).sum(1)
                if norm:
                    mean_emb = torch.nn.functional.normalize(mean_emb)

                embs.append(mean_emb)

        return torch.cat(embs, axis=0)

    def _mine_sentences(
        self, segment_embeddings: np.ndarray, input_embeddings: np.ndarray
    ):
        faiss.normalize_L2(segment_embeddings)
        faiss.normalize_L2(input_embeddings)

        knn = self._knnGPU if self.mining_device == "cuda" else self._knnCPU

        x2y_sim, x2y_ind = knn(
            segment_embeddings,
            input_embeddings,
            min(input_embeddings.shape[0], self.knn_neighborhood),
        )
        x2y_mean = x2y_sim.mean(axis=1)

        y2x_sim, y2x_ind = knn(
            input_embeddings,
            segment_embeddings,
            min(segment_embeddings.shape[0], self.knn_neighborhood),
        )
        y2x_mean = y2x_sim.mean(axis=1)

        if self.margin_algorithm == "absolute":
            margin = lambda a, b: a
        elif self.margin_algorithm == "distance":
            margin = lambda a, b: a - b
        else:  # args.margin == 'ratio':
            margin = lambda a, b: a / b

        fwd_scores = self._score_candidates(
            segment_embeddings,
            input_embeddings,
            x2y_ind,
            x2y_mean,
            y2x_mean,
            margin,
        )

        return self._get_pairs(fwd_scores, x2y_ind)

    def _get_manifest_lines(
        self, input_line: Dict[str, Any], segment_line: Dict[str, Any]
    ):
        ctm_path = segment_line["segments_level_ctm_filepath"]
        data = self._read_ctm_file(ctm_path)

        manifest = []

        segment_sents = []
        segment_pred_sents = []
        for i in data:
            segment_sents.append(i[2])
            segment_pred_sents.append(i[3])
            manifest.append(
                {
                    "course_id": input_line["course_id"],
                    "video_id": input_line["video_id"],
                    "text": i[2],
                    "pred_text": i[3],
                    "start_time": i[0],
                    "duration": i[1],
                    "alignment_score": self._calculate_alignment_score(i[2], i[3]),
                }
            )

        segment_embeddings = (
            self._encode_mean_pool(segment_sents, self.aligner_language_id)
            .cpu()
            .numpy()
        )

        for mining in input_line["text_mining"]:
            lang_id = mining["lang_id"]
            input_sents = self._read_text(mining["path"], mining["separators"])

            input_embeddings = (
                self._encode_mean_pool(input_sents, lang_id).cpu().numpy()
            )

            pairs: List[Tuple[int, int, float]] = self._mine_sentences(
                segment_embeddings, input_embeddings
            )

            for seg_idx, inp_idx, score in pairs:
                manifest[seg_idx].update(
                    {
                        f"{lang_id}_text": input_sents[inp_idx]
                        if inp_idx != -1
                        else "",
                        f"{lang_id}_sonar_score": score if inp_idx != -1 else 0,
                    }
                )

        return [json.dumps(line) for line in manifest]

    def run(self):
        with open(self.segment_manifest_path, "r") as fhand:
            segment_lines = fhand.readlines()

        with open(self.input_manifest_path, "r") as fhand:
            input_lines = fhand.readlines()

        # full_manifest = []
        for input_line, segment_line in tqdm(zip(input_lines, segment_lines)):
            manifest = self._get_manifest_lines(
                json.loads(input_line), json.loads(segment_line)
            )
            with open(
                f"{self.manifest_out_folder}/manifest_{self.language}.jsonl", "a+"
            ) as fhand:
                fhand.write("\n".join(manifest) + "\n")

    def cleanup(self):
        del self.encoder
        del self.tokenizer
