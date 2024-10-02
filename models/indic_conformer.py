from typing import List, Union

from base import BaseModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel


class IndicConformer(BaseModel):
    def __init__(
        self, model: Union[EncDecCTCModel, EncDecHybridRNNTCTCModel], language_id: str
    ):
        self.model = model
        self.language_id = language_id

    @property
    def tokenizer(self):
        return self.model.tokenizer.tokenizers_dict[self.language_id]

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def cfg(self):
        return self.model.cfg

    @property
    def preprocessor(self):
        return self.model.preprocessor

    def transcribe(
        self,
        audio_filepaths_batch: List[str],
        language_id: str,
        return_hypotheses: bool = False,
        batch_size: int = 1,
    ):
        return self.model.transcribe(
            audio_filepaths_batch,
            return_hypotheses=return_hypotheses,
            batch_size=batch_size,
            language_id=language_id,
        )
