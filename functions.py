import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def whisper(file_path):
   processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
   forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
   audio_data,sample_rate=librosa.load(file_path,sr=16000)
   input_features=processor(audio_data,sampling_rate=sample_rate,return_tensors='pt').input_features
   predicted_ids=model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
   transcription=processor.batch_decode(predicted_ids,skip_special_tokens=True)
   print(transcription[0])
   return transcription[0]


def mbart(trans):
   tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
   model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
   tokenizer.src_lang = "ko_KR"
   encoded = tokenizer(trans, return_tensors="pt")
   generated_tokens = model.generate(**encoded)
   trans2=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
   print(trans2)
   return trans2[0]
    
def load_tactron2_model():
    tactron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
    return tactron2
def load_hifigan():
   hifigan=HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
   return hifigan