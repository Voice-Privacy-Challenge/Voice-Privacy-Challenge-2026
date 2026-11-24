"""Text normalization for ASR transcripts.

Reference transcripts are fed to the attacker as they are. ASR transcripts are
brought to the convention of the text each head was trained on:

    none            identity
    upper_letters   upper case; letters, spaces and apostrophes only
                    (LibriSpeech convention, Track 1 head)
    lower_letters   lower case; letters, spaces, apostrophes and hyphens only
                    (MLS convention, Track 2 heads)
"""
import unicodedata


def normalize(text, mode='none'):
    if mode == 'none':
        return text
    text = unicodedata.normalize('NFC', text).replace('’', "'")
    if mode == 'upper_letters':
        # other characters become spaces, so that "well-known" stays two words
        text = ''.join(ch if ch.isalpha() or ch in " '" else ' ' for ch in text.upper())
    else:
        # other characters are deleted, as when the Track 2 training text was built
        text = ''.join(ch for ch in text.lower() if ch.isalpha() or ch in " '-")
    return ' '.join(text.split())
