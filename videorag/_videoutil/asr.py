

def speech_to_text(segment_index2name, segments):
    transcripts = {}
    for i, index in enumerate(segment_index2name):
        result = ""
        for segment in segments[i]['transcribe_audio']['chunks']:
            result += "[%.2fs -> %.2fs] %s\n" % (segment['start'], segment['end'], segment['text'])
        transcripts[index] = result
    
    return transcripts