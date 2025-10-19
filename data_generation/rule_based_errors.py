"""
Rule-based error generation for Nepali GEC
Implements 15+ error injection rules targeting common Nepali errors
"""
import random
import re
from typing import List, Tuple, Dict
import unicodedata

class NepaliErrorGenerator:
    """Generate grammatical errors in Nepali text using linguistic rules"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._initialize_error_patterns()
    
    def _initialize_error_patterns(self):
        """Initialize Nepali-specific error patterns"""
        
        # Vowel diacritics (raswa/dirgha confusion)
        self.vowel_pairs = [
            ('ा', 'ा'),  # Short a vs long aa
            ('ि', 'ी'),  # Short i vs long ii
            ('ु', 'ू'),  # Short u vs long uu
            ('े', 'ै'),  # e vs ai
            ('ो', 'ौ'),  # o vs au
        ]
        
        # Case markers (postpositions)
        self.case_markers = {
            'को': ['का', 'की', 'ले', 'लाई', 'मा'],  # Genitive
            'ले': ['को', 'लाई', 'मा', 'बाट'],  # Ergative/Instrumental
            'लाई': ['को', 'ले', 'मा'],  # Dative
            'मा': ['ले', 'को', 'बाट'],  # Locative
            'बाट': ['ले', 'मा', 'देखि'],  # Ablative
        }
        
        # Common verb endings for agreement errors
        self.verb_endings = {
            # Present tense
            'छ': ['छन्', 'छौ', 'छु'],  # 3rd sg -> 3rd pl, 2nd, 1st
            'छन्': ['छ', 'छौ', 'छु'],  # 3rd pl -> 3rd sg, 2nd, 1st
            'छु': ['छ', 'छौ', 'छन्'],  # 1st sg -> others
            # Past tense
            'थियो': ['थिए', 'थिइन्', 'थिएन'],
            'थिए': ['थियो', 'थिइन्', 'थिएन'],
        }
        
        # Honorific pronouns
        self.honorific_pronouns = {
            'तपाईं': ['तिमी', 'तँ'],  # Respectful -> casual/intimate
            'तिमी': ['तपाईं', 'तँ'],  # Casual -> respectful/intimate
            'उहाँ': ['उनी', 'ऊ'],  # He/she respectful -> casual
        }
        
        # Common animate/inanimate nouns for semantic errors
        self.animate_nouns = [
            'मान्छे', 'मानिस', 'केटा', 'केटी', 'बच्चा', 'शिक्षक', 
            'विद्यार्थी', 'डाक्टर', 'किसान', 'आमा', 'बुबा'
        ]
        
        self.inanimate_nouns = [
            'किताब', 'कलम', 'घर', 'कुर्सी', 'टेबल', 'गाडी',
            'झ्याल', 'ढोका', 'पानी', 'खाना', 'काम'
        ]
        
        # Agentive verbs (require animate subjects)
        self.agentive_verbs = [
            'खान्छ', 'पढ्छ', 'लेख्छ', 'बोल्छ', 'सोच्छ', 
            'हिँड्छ', 'गर्छ', 'भन्छ', 'सुन्छ'
        ]
    
    def generate_vowel_diacritic_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: ORTH:VOWEL (Raswa/Dirgha confusion)
        Swap short and long vowel diacritics
        """
        words = text.split()
        if len(words) < 3:
            return text, "NONE"
        
        # Find words with vowel diacritics
        candidates = []
        for i, word in enumerate(words):
            for short, long in self.vowel_pairs:
                if short in word or long in word:
                    candidates.append((i, word))
                    break
        
        if not candidates:
            return text, "NONE"
        
        idx, word = random.choice(candidates)
        error_word = word
        
        # Swap a vowel diacritic
        for short, long in self.vowel_pairs:
            if short in error_word:
                error_word = error_word.replace(short, long, 1)
                break
            elif long in error_word:
                error_word = error_word.replace(long, short, 1)
                break
        
        words[idx] = error_word
        return ' '.join(words), "ORTH:VOWEL"
    
    def generate_case_marker_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: MORPH:CASE
        Substitute incorrect case markers (postpositions)
        """
        words = text.split()
        
        # Find case markers
        for i, word in enumerate(words):
            if word in self.case_markers:
                # Replace with incorrect case marker
                incorrect_markers = self.case_markers[word]
                error_word = random.choice(incorrect_markers)
                words[i] = error_word
                return ' '.join(words), "MORPH:CASE"
        
        return text, "NONE"
    
    def generate_verb_agreement_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: MORPH:VERB_AGR
        Create subject-verb agreement violations
        """
        words = text.split()
        
        # Find verb endings
        for i, word in enumerate(words):
            for correct_ending, incorrect_endings in self.verb_endings.items():
                if word.endswith(correct_ending):
                    # Replace with incorrect agreement
                    wrong_ending = random.choice(incorrect_endings)
                    stem = word[:-len(correct_ending)]
                    words[i] = stem + wrong_ending
                    return ' '.join(words), "MORPH:VERB_AGR"
        
        return text, "NONE"
    
    def generate_spacing_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: ORTH:SPACING
        Add/remove spaces around postpositions
        """
        words = text.split()
        
        if len(words) < 2:
            return text, "NONE"
        
        # Find postpositions
        for i, word in enumerate(words):
            if word in self.case_markers and i > 0:
                # Attach postposition to previous word (remove space)
                words[i-1] = words[i-1] + words[i]
                words.pop(i)
                return ' '.join(words), "ORTH:SPACING"
        
        # Alternative: split a word incorrectly
        if len(words) > 0:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 4:
                split_point = len(word) // 2
                words[idx] = word[:split_point] + ' ' + word[split_point:]
                return ' '.join(words), "ORTH:SPACING"
        
        return text, "NONE"
    
    def generate_honorific_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: SEM:HONOR
        Create honorific level mismatches
        """
        words = text.split()
        
        for i, word in enumerate(words):
            if word in self.honorific_pronouns:
                # Replace with wrong honorific level
                incorrect_pronouns = self.honorific_pronouns[word]
                words[i] = random.choice(incorrect_pronouns)
                return ' '.join(words), "SEM:HONOR"
        
        return text, "NONE"
    
    def generate_semantic_selectional_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: SEM:SELECT
        Create selectional preference violations (e.g., inanimate subject with agentive verb)
        """
        words = text.split()
        
        # Find agentive verbs
        has_agentive = False
        for word in words:
            if any(verb in word for verb in self.agentive_verbs):
                has_agentive = True
                break
        
        if has_agentive:
            # Replace subject with inanimate noun
            if len(words) > 1:
                words[0] = random.choice(self.inanimate_nouns)
                return ' '.join(words), "SEM:SELECT"
        
        return text, "NONE"
    
    def generate_semantic_entity_type_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: SEM:ENTITY
        Swap animate and inanimate entities creating semantic implausibility
        """
        words = text.split()
        
        # Find animate nouns and replace with inanimate (or vice versa)
        for i, word in enumerate(words):
            if word in self.animate_nouns:
                words[i] = random.choice(self.inanimate_nouns)
                return ' '.join(words), "SEM:ENTITY"
            elif word in self.inanimate_nouns:
                words[i] = random.choice(self.animate_nouns)
                return ' '.join(words), "SEM:ENTITY"
        
        return text, "NONE"
    
    def generate_conjunct_error(self, text: str) -> Tuple[str, str]:
        """
        Error Type: ORTH:CONJUNCT
        Introduce errors in conjunct consonants (halant usage)
        """
        # Complex pattern - simplified version
        words = text.split()
        
        # Look for potential conjunct positions
        for i, word in enumerate(words):
            if '्' in word:  # Has halant
                # Randomly remove or duplicate halant
                if random.random() < 0.5:
                    words[i] = word.replace('्', '', 1)
                else:
                    # Add extra halant
                    pos = word.find('्')
                    if pos > 0:
                        words[i] = word[:pos] + '्' + word[pos:]
                return ' '.join(words), "ORTH:CONJUNCT"
        
        return text, "NONE"
    
    def generate_error(self, text: str, error_type: str = None) -> Tuple[str, str]:
        """
        Generate a specific type of error or random error
        
        Args:
            text: Clean Nepali sentence
            error_type: Specific error type or None for random
        
        Returns:
            (erroneous_text, error_type_code)
        """
        error_functions = {
            "ORTH:VOWEL": self.generate_vowel_diacritic_error,
            "MORPH:CASE": self.generate_case_marker_error,
            "MORPH:VERB_AGR": self.generate_verb_agreement_error,
            "ORTH:SPACING": self.generate_spacing_error,
            "SEM:HONOR": self.generate_honorific_error,
            "SEM:SELECT": self.generate_semantic_selectional_error,
            "SEM:ENTITY": self.generate_semantic_entity_type_error,
            "ORTH:CONJUNCT": self.generate_conjunct_error,
        }
        
        if error_type and error_type in error_functions:
            return error_functions[error_type](text)
        else:
            # Random error type
            func = random.choice(list(error_functions.values()))
            result = func(text)
            # If no error generated, try another
            if result[1] == "NONE":
                for func in error_functions.values():
                    result = func(text)
                    if result[1] != "NONE":
                        return result
            return result
    
    def generate_multi_error(self, text: str, num_errors: int = 2) -> Tuple[str, List[str]]:
        """
        Generate multiple errors in a single sentence
        
        Returns:
            (erroneous_text, list_of_error_types)
        """
        error_types = []
        current_text = text
        
        for _ in range(num_errors):
            current_text, error_type = self.generate_error(current_text)
            if error_type != "NONE":
                error_types.append(error_type)
        
        return current_text, error_types


# Example usage and testing
if __name__ == "__main__":
    generator = NepaliErrorGenerator()
    
    # Test sentences
    test_sentences = [
        "म किताब पढ्छु",
        "उनी घर मा छन्",
        "तपाईं कहाँ जानुहुन्छ",
        "बच्चा खाना खान्छ",
        "शिक्षक विद्यालय मा पढाउँछन्",
    ]
    
    print("=== Rule-Based Error Generation Examples ===\n")
    
    for sent in test_sentences:
        error_text, error_type = generator.generate_error(sent)
        print(f"Original: {sent}")
        print(f"Error:    {error_text}")
        print(f"Type:     {error_type}")
        print()
    
    print("\n=== Multi-Error Generation ===\n")
    for sent in test_sentences[:2]:
        error_text, error_types = generator.generate_multi_error(sent, num_errors=2)
        print(f"Original: {sent}")
        print(f"Errors:   {error_text}")
        print(f"Types:    {', '.join(error_types)}")
        print()