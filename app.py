import argparse
import logging

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri

# Imports needed for distil whisper
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Import needs for processing video file
import tempfile
import ffmpeg
import torchaudio


def process_output_with_enclosed_timestamps(output):
        segments = []
        raw_segments = output[0].split("<|")
        for i in range(1, len(raw_segments), 2):  
            start_segment = raw_segments[i].split("|>")
            end_segment = raw_segments[i+1].split("|>") if i+1 < len(raw_segments) else ["", ""]
            if len(start_segment) == 2 and len(end_segment) >= 1:
                start_timestamp = int(float(start_segment[0]) * 1000)
                text = start_segment[1]
                end_timestamp = int(float(end_segment[0]) * 1000)
                segments.append((start_timestamp, end_timestamp, text))
        return segments

def audio_duration_is_long(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        duration = float(probe['streams'][0]['duration'])
        if duration > 30:
            return True
        else:
            return False
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
        return False

class DistilWhisperWrapper(ClamsApp):

    model_size_alias = {
        'small': 'distil-small.en', 
        's': 'distil-small.en', 
        'medium': 'distil-medium.en', 
        'm': 'distil-medium.en', 
        'large-v2': 'distil-large-v2', 
        'l2': 'distil-large-v2', 
        'large-v3': 'distil-large-v3',
        'l3': 'distil-large-v3'
    }

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        if not isinstance(mmif, Mmif):
            mmif: Mmif = Mmif(mmif)

        # prepare the proper model name
        size = parameters['modelSize']
        if size in self.model_size_alias:
            size = self.model_size_alias[size]
        self.logger.debug(f'distil whisper model: {size})')

        # prepare the distil model
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = f'distil-whisper/{size}'
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        # get the file path
        docs = mmif.get_documents_by_type(DocumentTypes.AudioDocument)
        if docs:
            doc = docs[0]
            target_path = doc.location_path(nonexist_ok=False)
        else:
            docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
            doc = docs[0]
            video_path = doc.location_path(nonexist_ok=False)
            # transform the video file to audio file
            audio_tmpdir = tempfile.TemporaryDirectory()
            target_path = f'{audio_tmpdir.name}/{doc.id}_16kHz.wav'
            ffmpeg.input(video_path).output(target_path, ac=1, ar=16000).run()
        
        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument, document=doc.long_id)
        new_view.new_contain(AnnotationTypes.TimeFrame, timeUnit="milliseconds", document=doc.long_id)
        new_view.new_contain(AnnotationTypes.Alignment)
        new_view.new_contain(Uri.SENTENCE, document=doc.long_id)

        # model run on long form audio using the model + processor API directly
        if audio_duration_is_long(target_path):
            # process the audio into tensor
            waveform, sampling_rate = torchaudio.load(target_path)
            if sampling_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                    waveform = resampler(waveform)
                    sampling_rate = 16000

            inputs = processor(
                waveform.squeeze().numpy(),  # Convert to numpy array and remove channel dimension if it's single-channel audio
                sampling_rate=sampling_rate,
                return_tensors="pt",  # Return PyTorch tensors
                padding="longest",  # This might not be necessary for a single file, but it's here for consistency
                return_attention_mask=True,
                truncation=False
            )
            generate_kwargs = {
                "max_new_tokens": 448,
                "num_beams": 1,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "return_timestamps": True,
            }
            inputs = inputs.to(device, dtype=torch_dtype)
            pred_ids = model.generate(**inputs, **generate_kwargs)
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=True)
            processed_segments = process_output_with_enclosed_timestamps(pred_text)

            td = ''.join(text for _, _, text in processed_segments)[1:]
            text_document: Document = new_view.new_textdocument(text=td)
            new_view.new_annotation(AnnotationTypes.Alignment, source=doc.long_id, target=text_document.long_id)
            for segment in processed_segments:
                sentence = new_view.new_annotation(Uri.SENTENCE, text=segment[2][1:])
                tf = new_view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=segment[0], end=segment[1])
                new_view.new_annotation(AnnotationTypes.Alignment, source=tf.long_id, target=sentence.long_id)
        
        # model run on short form using pipeline
        else:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=torch_dtype,
                device=device,
            )
            result = pipe(target_path, return_timestamps=True)
            output_text = result["text"]
            text_document: Document = new_view.new_textdocument(text=output_text)
            new_view.new_annotation(AnnotationTypes.Alignment, source=doc.long_id, target=text_document.long_id)
            for chunk in result["chunks"]:
                sentence = new_view.new_annotation(Uri.SENTENCE, text=chunk['text'])
                time = chunk["timestamp"]
                s = int(time[0] * 1000)
                e = int(time[1] * 1000)
                tf = new_view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=s, end=e)
                new_view.new_annotation(AnnotationTypes.Alignment, source=tf.long_id, target=sentence.long_id)
        
        return mmif



def get_app():
    return DistilWhisperWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
