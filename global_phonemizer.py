
import phonemizer

class Phonemizer:
    def __init__(self, lang):
        self.phonemizer = self.setup_phonemizer(lang)
    def setup_phonemizer(self, language):
        """Setup phonemizer for a specific language if not already initialized"""
        return phonemizer.backend.EspeakBackend(
                language=language, preserve_punctuation=True, with_stress=True
            )
    def phonemize(self, text):
        ps = self.phonemizer.phonemize([text])[0]
        return ps.strip()
    
if __name__ == "__main__":
    phonemizer = Phonemizer("sw")
    print(phonemizer.phonemize("Mungu alimwambia kuwa yeye ni Mungu!"))