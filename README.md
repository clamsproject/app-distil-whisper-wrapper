# Distil-Whisper Wrapper

## Description
Wrapper for Distil-Whisper https://github.com/huggingface/distil-whisper

## Input
The wrapper takes miff file which refers to either an AudioDocument or a VideoDocument. You can choose the
specific model you want to use by add parameter. Four models are available: 'distil-large-v3', 'distil-large-v2', 'distil-medium.en', 'distil-small.en'. The default model is 'distil-small.en'.

## Output
The output miff file will contain three objects: **Uri.SENTENCE**, which is the smallest text unit recognized by the distil-whisper; **AnnotationTypes.TimeFrame**, which represents the timeframe that each sentences take place; **DocumentTypes.TextDocument**, which contains all the text recognized by distil-whisper in the whole audio/video; **AnnotationTypes.BoundingBox**, which show the alignments between `Timeframe` <-> `SENTENCE`, and `auodio/video` <-> `TextDocument`.

## User instruction
General user instructions for CLAMS apps are available at CLAMS Apps documentation: https://apps.clams.ai/clamsapp/

### System requirements
- Requires **numpy<2**
- Requires python library **transformers** and **accelerate**
- Requires **ffmpeg-python** for the VideoDocument
