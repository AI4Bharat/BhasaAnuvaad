import copy
import importlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import glob2
import torch
from base import BaseStep
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.utils import logging
from omegaconf import OmegaConf
from step_decorator import Step
from utils.data_prep import (
    add_t_start_end_to_utt_obj,
    get_batch_starts_ends,
    get_batch_variables,
    get_manifest_lines_batch,
    is_entry_in_all_lines,
    is_entry_in_any_lines,
)
from utils.make_ass_files import make_ass_files
from utils.make_ctm_files import make_ctm_files
from utils.make_output_manifest import write_manifest_out_line
from utils.viterbi_decoding import viterbi_decoding


@dataclass
class CTMFileConfig:
    remove_blank_tokens: bool = False
    # minimum duration (in seconds) for timestamps in the CTM.If any line in the CTM has a
    # duration lower than this, it will be enlarged from the middle outwards until it
    # meets the minimum_timestamp_duration, or reaches the beginning or end of the audio file.
    # Note that this may cause timestamps to overlap.
    minimum_timestamp_duration: float = 0


@dataclass
class ASSFileConfig:
    fontsize: int = 20
    vertical_alignment: str = "center"
    # if resegment_text_to_fill_space is True, the ASS files will use new segments
    # such that each segment will not take up more than (approximately) max_lines_per_segment
    # when the ASS file is applied to a video
    resegment_text_to_fill_space: bool = False
    max_lines_per_segment: int = 2
    text_already_spoken_rgb: List[int] = field(
        default_factory=lambda: [49, 46, 61]
    )  # dark gray
    text_being_spoken_rgb: List[int] = field(
        default_factory=lambda: [57, 171, 9]
    )  # dark green
    text_not_yet_spoken_rgb: List[int] = field(
        default_factory=lambda: [194, 193, 199]
    )  # light gray


@dataclass
class AlignmentConfig:
    # Required configs
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    manifest_filepath: Optional[str] = None
    output_dir: Optional[str] = None
    language_id: Optional[str] = None

    # General configs
    align_using_pred_text: bool = False
    transcribe_device: Optional[str] = None
    viterbi_device: Optional[str] = None
    manifest_batch_size: int = 1
    use_silero_vad: bool = False
    vad_chunked_batch_size: int = 1
    use_local_attention: bool = True
    additional_segment_grouping_separator: Optional[str] = None
    audio_filepath_parts_in_utt_id: int = 1

    # Buffered chunked streaming configs
    use_buffered_chunked_streaming: bool = False
    chunk_len_in_secs: float = 1.6
    total_buffer_in_secs: float = 4.0
    chunk_batch_size: int = 32

    # Cache aware streaming configs
    simulate_cache_aware_streaming: Optional[bool] = False

    # Output file configs
    save_output_file_formats: List[str] = field(default_factory=lambda: ["ctm"])
    ctm_file_config: CTMFileConfig = field(default_factory=lambda: CTMFileConfig())
    ass_file_config: ASSFileConfig = field(default_factory=lambda: ASSFileConfig())


@Step("aligner")
class Aligner(BaseStep):
    def initialise(
        self,
        infra: Dict[str, int],
        model_path: Optional[str] = None,
        pretrained_name: Optional[str] = None,
        additional_segment_grouping_separator: Optional[str] = None,
        custom_model_class: Optional[str] = None,
        align_using_pred_text: bool = False,
        transcribe_device: Optional[str] = None,
        viterbi_device: Optional[str] = None,
        manifest_batch_size: int = 1,
        use_silero_vad: bool = False,
        vad_chunked_batch_size: int = 1,
    ):
        self.infra = infra

        aligner_manifest_path = self.get_state("aligner_manifest_path")
        language = self.get_state("aligner_language")
        manifest_filepath = f"{aligner_manifest_path}/manifest_{language}.jsonl"

        output_dir = self.get_state("aligned_output_path") + f"/{language}"

        self.done = list(glob2.iglob(f"{output_dir}/ctm/segments/*.ctm"))
        self.cfg = AlignmentConfig(
            pretrained_name=pretrained_name,
            model_path=model_path,
            manifest_filepath=manifest_filepath,
            output_dir=output_dir,
            language_id=self.get_state("aligner_language_id"),
            align_using_pred_text=align_using_pred_text,
            transcribe_device=transcribe_device,
            viterbi_device=viterbi_device,
            manifest_batch_size=manifest_batch_size,
            use_silero_vad=use_silero_vad,
            vad_chunked_batch_size=vad_chunked_batch_size,
            use_local_attention=True,
            additional_segment_grouping_separator=additional_segment_grouping_separator,
        )

        # Validate config
        if self.cfg.model_path is None and self.cfg.pretrained_name is None:
            raise ValueError("Both model_path and pretrained_name cannot be None")

        if self.cfg.model_path is not None and self.cfg.pretrained_name is not None:
            raise ValueError("One of model_path and pretrained_name must be None")

        if self.cfg.manifest_batch_size < 1:
            raise ValueError("manifest_batch_size cannot be zero or a negative number")

        if self.cfg.vad_chunked_batch_size < 1:
            raise ValueError(
                "vad_chunked_batch_size cannot be zero or a negative number"
            )

        if (
            self.cfg.additional_segment_grouping_separator == ""
            or self.cfg.additional_segment_grouping_separator == " "
        ):
            raise ValueError(
                "additional_grouping_separator cannot be empty string or space character"
            )

        if self.cfg.ctm_file_config.minimum_timestamp_duration < 0:
            raise ValueError("minimum_timestamp_duration cannot be a negative number")

        if self.cfg.ass_file_config.vertical_alignment not in [
            "top",
            "center",
            "bottom",
        ]:
            raise ValueError(
                "ass_file_config.vertical_alignment must be one of 'top', 'center' or 'bottom'"
            )

        for rgb_list in [
            self.cfg.ass_file_config.text_already_spoken_rgb,
            self.cfg.ass_file_config.text_already_spoken_rgb,
            self.cfg.ass_file_config.text_already_spoken_rgb,
        ]:
            if len(rgb_list) != 3:
                raise ValueError(
                    "ass_file_config.text_already_spoken_rgb,"
                    " ass_file_config.text_being_spoken_rgb,"
                    " and ass_file_config.text_already_spoken_rgb all need to contain"
                    " exactly 3 elements."
                )

        # init devices
        if self.cfg.transcribe_device is None:
            self.transcribe_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.transcribe_device = torch.device(self.cfg.transcribe_device)
        logging.info(
            f"Device to be used for transcription step (`transcribe_device`) is {self.transcribe_device}"
        )

        if self.cfg.viterbi_device is None:
            self.viterbi_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.viterbi_device = torch.device(self.cfg.viterbi_device)
        logging.info(
            f"Device to be used for viterbi step (`viterbi_device`) is {self.viterbi_device}"
        )

        if self.transcribe_device.type == "cuda" or self.viterbi_device.type == "cuda":
            logging.warning(
                "One or both of transcribe_device and viterbi_device are GPUs. If you run into OOM errors "
                "it may help to change both devices to be the CPU."
            )

        # load model
        self.model, _ = setup_model(self.cfg, self.transcribe_device)
        self.model.eval()

        if isinstance(self.model, EncDecHybridRNNTCTCModel):
            self.model.change_decoding_strategy(decoder_type="ctc")

        if self.cfg.use_local_attention:
            logging.info(
                "Flag use_local_attention is set to True => will try to use local attention for model if it allows it"
            )
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn", att_context_size=[64, 64]
            )

        if not (
            isinstance(self.model, EncDecCTCModel)
            or isinstance(self.model, EncDecHybridRNNTCTCModel)
        ):
            raise NotImplementedError(
                "Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
                " Currently only instances of these models are supported"
            )

        if self.cfg.ctm_file_config.minimum_timestamp_duration > 0:
            logging.warning(
                f"ctm_file_config.minimum_timestamp_duration has been set to {self.cfg.ctm_file_config.minimum_timestamp_duration} seconds. "
                "This may cause the alignments for some tokens/words/additional segments to be overlapping."
            )

        self.buffered_chunk_params = {}
        if self.cfg.use_buffered_chunked_streaming:
            model_cfg = copy.deepcopy(self.model._cfg)

            OmegaConf.set_struct(model_cfg.model_preprocessor, False)
            # some changes for streaming scenario
            model_cfg.model_preprocessor.dither = 0.0
            model_cfg.model_preprocessor.pad_to = 0

            if model_cfg.model_preprocessor.normalize != "per_feature":
                logging.error(
                    "Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently"
                )
            # Disable config overwriting
            OmegaConf.set_struct(model_cfg.model_preprocessor, True)

            feature_stride = model_cfg.model_preprocessor["window_stride"]
            model_stride_in_secs = feature_stride * self.cfg.model_downsample_factor
            total_buffer = self.cfg.total_buffer_in_secs
            chunk_len = float(self.cfg.chunk_len_in_secs)
            tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
            mid_delay = math.ceil(
                (chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs
            )
            logging.info(
                f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}"
            )

            self.model = FrameBatchASR(
                asr_model=self.model,
                frame_len=chunk_len,
                total_buffer=self.cfg.total_buffer_in_secs,
                batch_size=self.cfg.chunk_batch_size,
            )
            self.buffered_chunk_params = {
                "delay": mid_delay,
                "model_stride_in_secs": model_stride_in_secs,
                "tokens_per_chunk": tokens_per_chunk,
            }
        elif custom_model_class is not None:
            module = importlib.import_module("models")

            # Get the class from the module
            model_cls = getattr(module, custom_model_class)
            self.model = model_cls(
                model=self.model, language_id=self.get_state("aligner_language_id")
            )

        # get start and end line IDs of batches
        self.starts, self.ends = get_batch_starts_ends(
            manifest_filepath, manifest_batch_size
        )

        # init output_timestep_duration = None and we will calculate and update it during the first batch
        self.output_timestep_duration = None

        # init f_manifest_out
        tgt_manifest_name = (
            str(Path(manifest_filepath).stem) + "_with_output_file_paths.json"
        )
        os.makedirs(output_dir, exist_ok=True)
        tgt_manifest_filepath = str(Path(output_dir) / tgt_manifest_name)
        self.f_manifest_out = open(tgt_manifest_filepath, "a+")

    def run(self) -> Any:
        # get alignment and save in CTM batch-by-batch
        for start, end in zip(self.starts, self.ends):
            manifest_lines_batch = get_manifest_lines_batch(
                self.cfg.manifest_filepath, start, end, self.cfg.language_id
            )

            done = sum(
                [
                    f'{self.cfg.output_dir}/ctm/segments/{line["audio_filepath"].split("/")[-1][:-4]}.ctm'
                    in self.done
                    for line in manifest_lines_batch
                ]
            )
            if done == self.cfg.manifest_batch_size:
                continue

            (
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                utt_obj_batch,
                output_timestep_duration,
            ) = get_batch_variables(
                manifest_lines_batch,
                self.model,
                self.cfg.language_id,
                self.cfg.additional_segment_grouping_separator,
                self.cfg.align_using_pred_text,
                self.cfg.audio_filepath_parts_in_utt_id,
                self.output_timestep_duration,
                self.cfg.use_silero_vad,
                self.cfg.vad_chunked_batch_size,
                self.cfg.simulate_cache_aware_streaming,
                self.cfg.use_buffered_chunked_streaming,
                self.buffered_chunk_params,
            )

            alignments_batch = viterbi_decoding(
                log_probs_batch, y_batch, T_batch, U_batch, self.viterbi_device
            )

            for idx, (utt_obj, alignment_utt) in enumerate(
                zip(utt_obj_batch, alignments_batch)
            ):
                utt_obj = add_t_start_end_to_utt_obj(
                    utt_obj, alignment_utt, output_timestep_duration
                )

                if "ctm" in self.cfg.save_output_file_formats:
                    utt_obj = make_ctm_files(
                        utt_obj,
                        self.cfg.output_dir,
                        self.cfg.ctm_file_config,
                        self.model,
                        log_probs_batch[idx],
                        output_timestep_duration,
                    )

                if "ass" in self.cfg.save_output_file_formats:
                    utt_obj = make_ass_files(
                        utt_obj, self.cfg.output_dir, self.cfg.ass_file_config
                    )

                write_manifest_out_line(
                    self.f_manifest_out,
                    utt_obj,
                )

        self.f_manifest_out.close()

    def cleanup(self):
        del self.model
