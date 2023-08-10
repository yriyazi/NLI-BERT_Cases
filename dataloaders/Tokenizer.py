from parsinorm import General_normalization,Special_numbers, \
                        Abbreviation , TTS_normalization

class prepos():
    def __init__(self) -> None:
        self.general_normalization  = General_normalization()
        self.abbreviation           = Abbreviation()
        self.special_numbers        = Special_numbers()
        self.TTS_normalization      = TTS_normalization()
        
    def forward(self,sentence:str):
        # Normalization
        out = self.general_normalization.alphabet_correction(      sentence)
        out = self.general_normalization.semi_space_correction(         out)
        out = self.general_normalization.arabic_correction(             out)
        out = self.general_normalization.punctuation_correction(        out)
        out = self.general_normalization.specials_chars(                out)
        out = self.general_normalization.remove_emojis(                 out)
        out = self.general_normalization.remove_not_desired_chars(      out)
        
        # replacing abbreviations to words
        out = self.abbreviation.replace_date_abbreviation(              out)
        out = self.abbreviation.replace_persian_label_abbreviation(     out)
        out = self.abbreviation.replace_law_abbreviation(               out)
        out = self.abbreviation.replace_book_abbreviation(              out)
        out = self.abbreviation.replace_other_abbreviation(             out)
        out = self.abbreviation.replace_English_abbrevations(           out)
        
        # replacing numbers to words mathamtically corect
        out = self.TTS_normalization.math_correction(                   out)
        out = self.TTS_normalization.replace_currency(                  out)
        out = self.TTS_normalization.replace_symbols(                   out)
    
        # replacing numbers to words mathamtically corect
        out = self.special_numbers.replace_national_code(               out)
        out = self.special_numbers.convert_numbers_to_text(             out)
        return out
    
            
