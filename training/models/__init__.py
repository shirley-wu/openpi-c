from transformers import (
    BartConfig, BartTokenizer, BartForConditionalGeneration,
)

from .bart_emem import BartForConditionalGenerationEMem

MODEL_CLASSES = {
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
    "bart-emem": (BartConfig, BartForConditionalGenerationEMem, BartTokenizer),
}
