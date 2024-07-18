"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from lapps.discriminators import Uri


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Distil Whisper Wrapper",
        description="The wrapper of Distil-Whisper, avaliable models: distil-large-v3, distil-large-v2, distil-medium.en, distil-small.en. The default model is distil-small.en.",  # briefly describe what the purpose and features of the app
        app_license="Apache 2.0",  # short name for a software license like MIT, Apache2, GPL, etc.
        identifier="distil-whisper-wrapper",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format.
        url="https://github.com/clamsproject/app-distil-whisper-wrapper",  # a website where the source code and full documentation of the app is hosted
        analyzer_version='1.0', # use this IF THIS APP IS A WRAPPER of an existing computational analysis algorithm
        analyzer_license="MIT",  # short name for a software license
    )
    # and then add I/O specifications: an app must have at least one input and one output
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    out_td = metadata.add_output(DocumentTypes.TextDocument, **{'@lang': 'en'})
    out_td.add_description('Fully serialized text content of the recognized text in the input audio/video.')
    timeunit = "milliseconds"
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit=timeunit)
    out_ali = metadata.add_output(AnnotationTypes.Alignment)
    out_ali.add_description('Alignments between 1) `TimeFrame` <-> `SENTENCE`, 2) `audio/video document` <-> `TextDocument`')
    out_sent = metadata.add_output(Uri.SENTENCE)
    out_sent.add_description('The smallest recognized unit of distil-whisper. Normally a complete sentence.')
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(
        name='modelSize', 
        description='The size of the model to use. There are four size of model to use distil-large-v3, distil-large-v2, distil-medium.en, distil-small.en. You can also enter the abbreviation of the model as parameter. \'small\' and \'s\' for distil-small.en; \'medium\' and  \'m\' for distil-medium.en; \'large-v2\' and \'l2\' for distil-large-v2; \'large-v3\' and \'l3\' for distil-large-v3. The default model is distil-medium.en.)',
        type='string',
        choices=['distil-large-v3', 'distil-large-v2', 'distil-medium.en', 'distil-small.en', 'small', 's', 'medium', 'm', 'large-v2', 'l2', 'large-v3', 'l3'],
        default="distil-small.en"
    )
    # metadta.add_parameter(more...)
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
