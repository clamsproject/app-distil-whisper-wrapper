"""
DELETE THIS MODULE STRING AND REPLACE IT WITH A DESCRIPTION OF YOUR APP.

app.py Template

The app.py script does several things:
- import the necessary code
- create a subclass of ClamsApp that defines the metadata and provides a method to run the wrapped NLP tool
- provide a way to run the code as a RESTful Flask service


"""

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
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

        # try to get AudioDocuments
        docs = mmif.get_documents_by_type(DocumentTypes.AudioDocument)
        if docs:
            for doc in docs:
                file_path = doc.location_path(nonexist_ok=False)
                new_view = mmif.new_view()
                self.sign_view(new_view, parameters)
                new_view.new_contain(DocumentTypes.TextDocument, document=doc.long_id)
                new_view.new_contain(AnnotationTypes.TimeFrame, timeUnit="milliseconds", document=doc.long_id)
                new_view.new_contain(AnnotationTypes.Alignment)

                result = pipe(file_path, return_timestamps=True)
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


        # and if none found, try VideoDocuments
        else:
            docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
            for doc in docs:
                video_path = doc.location_path(nonexist_ok=False)
                # transform the video file to audio file
                audio_tmpdir = tempfile.TemporaryDirectory()
                resampled_audio_fname = f'{audio_tmpdir.name}/{doc.id}_16kHz.wav'
                ffmpeg.input(video_path).output(resampled_audio_fname, ac=1, ar=16000).run()

                new_view = mmif.new_view()
                self.sign_view(new_view, parameters)
                new_view.new_contain(DocumentTypes.TextDocument, document=doc.long_id)
                new_view.new_contain(AnnotationTypes.TimeFrame, timeUnit="milliseconds", document=doc.long_id)
                new_view.new_contain(AnnotationTypes.Alignment)

                result = pipe(resampled_audio_fname, return_timestamps=True)
                output_text = result["text"]
                text_document: Document = new_view.new_textdocument(text=output_text)
                new_view.new_annotation(AnnotationTypes.Alignment, source=doc.long_id, target=text_document.long_id)
                for chunk in result["chunks"]:
                    sentence = new_view.new_annotation(Uri.SENTENCE, text=chunk['text'])
                    time = chunk["timestamp"]
                    s = int(time[0] * 1000)
                    e = int(time[1] * 1000)
                    tf = new_view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", timeUnit="milliseconds", start=s, end=e)
                    new_view.new_annotation(AnnotationTypes.Alignment, source=tf.long_id, target=sentence.long_id)
        return mmif



def get_app():
    return DistilWhisperWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

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
