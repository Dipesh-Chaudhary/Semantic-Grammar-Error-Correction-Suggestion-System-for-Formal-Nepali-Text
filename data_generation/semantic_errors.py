"""
Advanced Semantic Error Generation for Nepali
Creates semantically implausible sentences while maintaining grammatical correctness
"""
import random
from typing import List, Tuple, Dict

class SemanticErrorGenerator:
    """Generate semantic errors in Nepali text"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._initialize_semantic_patterns()
    
    def _initialize_semantic_patterns(self):
        """Initialize semantic categories and relationships"""
        
        # Animate entities (agents)
        self.human_entities = [
            'मान्छे', 'मानिस', 'केटा', 'केटी', 'बच्चा', 'बालक',
            'शिक्षक', 'डाक्टर', 'किसान', 'विद्यार्थी', 'नर्स',
            'आमा', 'बुबा', 'दाजु', 'दिदी', 'साथी', 'छोरा', 'छोरी'
        ]
        
        self.animal_entities = [
            'कुकुर', 'बिरालो', 'गाई', 'भैंसी', 'बाख्रा', 'भेडा',
            'चरा', 'हात्ती', 'घोडा', 'माछा'
        ]
        
        # Inanimate entities (patients/themes)
        self.concrete_objects = [
            'किताब', 'कलम', 'कागज', 'झोला', 'घडी', 'मोबाइल',
            'कुर्सी', 'टेबल', 'झ्याल', 'ढोका', 'बत्ती', 'भाँडो'
        ]
        
        self.abstract_concepts = [
            'विचार', 'सपना', 'आशा', 'डर', 'खुशी', 'दुःख',
            'समय', 'स्वतन्त्रता', 'शान्ति', 'प्रेम'
        ]
        
        self.food_items = [
            'भात', 'दाल', 'तरकारी', 'रोटी', 'दूध', 'पानी',
            'फलफूल', 'मासु', 'खाना', 'चिया'
        ]
        
        self.places = [
            'घर', 'विद्यालय', 'बजार', 'अस्पताल', 'पार्क',
            'मन्दिर', 'गाउँ', 'शहर', 'देश'
        ]
        
        # Verb classifications by semantic selectional restrictions
        
        # Verbs requiring ANIMATE subjects (agentive)
        self.agentive_verbs = {
            'खान्छ': ['खान्छ', 'खान्छन्', 'खान्छिन्'],  # eat
            'पढ्छ': ['पढ्छ', 'पढ्छन्', 'पढ्छिन्'],  # read/study
            'लेख्छ': ['लेख्छ', 'लेख्छन्', 'लेख्छिन्'],  # write
            'बोल्छ': ['बोल्छ', 'बोल्छन्', 'बोल्छिन्'],  # speak
            'सोच्छ': ['सोच्छ', 'सोच्छन्', 'सोच्छिन्'],  # think
            'हिँड्छ': ['हिँड्छ', 'हिँड्छन्', 'हिँड्छिन्'],  # walk
            'दौडन्छ': ['दौडन्छ', 'दौडन्छन्', 'दौडन्छिन्'],  # run
            'रुन्छ': ['रुन्छ', 'रुन्छन्', 'रुन्छिन्'],  # cry
            'हाँस्छ': ['हाँस्छ', 'हाँस्छन्', 'हाँस्छिन्'],  # laugh
        }
        
        # Verbs requiring HUMAN subjects (cognitive/communication)
        self.human_verbs = {
            'सिक्छ': ['सिक्छ', 'सिक्छन्', 'सिक्छिन्'],  # learn
            'सिकाउँछ': ['सिकाउँछ', 'सिकाउँछन्', 'सिकाउँछिन्'],  # teach
            'बुझ्छ': ['बुझ्छ', 'बुझ्छन्', 'बुझ्छिन्'],  # understand
            'भन्छ': ['भन्छ', 'भन्छन्', 'भन्छिन्'],  # say
        }
        
        # Verbs requiring EDIBLE objects
        self.consumption_verbs = {
            'खान्छ': self.food_items,  # eat
            'पिउँछ': ['पानी', 'दूध', 'चिया', 'जुस'],  # drink
        }
        
        # Verbs requiring CONCRETE objects
        self.manipulation_verbs = {
            'उठाउँछ': self.concrete_objects,  # lift/pick up
            'बोक्छ': self.concrete_objects + self.food_items,  # carry
            'फ्याँक्छ': self.concrete_objects,  # throw
        }
        
        # Honorific context patterns
        self.formal_pronouns = ['तपाईं', 'हजुर']
        self.informal_pronouns = ['तिमी', 'तँ']
        self.formal_verb_endings = ['हुनुहुन्छ', 'गर्नुहुन्छ', 'हुनुभयो']
        self.informal_verb_endings = ['छस्', 'गर्छस्', 'भयस्']
    
    def create_animate_inanimate_violation(self, sentence: str) -> Tuple[str, str]:
        """
        Create error: Inanimate subject with agentive verb
        Example: "किताबले खाना खान्छ" (Book eats food) - grammatical but semantically wrong
        """
        words = sentence.split()
        
        # Find agentive verb
        verb_found = None
        for word in words:
            for base_verb in self.agentive_verbs.keys():
                if base_verb in word:
                    verb_found = base_verb
                    break
            if verb_found:
                break
        
        if verb_found and len(words) > 0:
            # Replace subject with inanimate object
            inanimate = random.choice(self.concrete_objects + self.abstract_concepts)
            words[0] = inanimate
            return ' '.join(words), "SEM:ANIM_VIOLATION"
        
        return sentence, "NONE"
    
    def create_object_type_violation(self, sentence: str) -> Tuple[str, str]:
        """
        Create error: Wrong object type for verb
        Example: "म घर खान्छु" (I eat house) - grammatically OK but semantically impossible
        """
        words = sentence.split()
        
        # Look for consumption verbs
        for verb, valid_objects in self.consumption_verbs.items():
            if any(verb in word for word in words):
                # Replace object with invalid type
                invalid_object = random.choice(self.concrete_objects + self.places)
                # Try to find object position (usually before verb)
                if len(words) > 2:
                    words[-2] = invalid_object
                    return ' '.join(words), "SEM:OBJ_TYPE"
        
        return sentence, "NONE"
    
    def create_honorific_context_violation(self, sentence: str) -> Tuple[str, str]:
        """
        Create error: Formal pronoun with informal verb or vice versa
        Example: "तपाईं खान्छस्" (You[formal] eat[informal]) - pragmatically wrong
        """
        words = sentence.split()
        
        # Find pronoun
        has_formal = any(p in words for p in self.formal_pronouns)
        has_informal = any(p in words for p in self.informal_pronouns)
        
        if has_formal:
            # Add informal verb ending
            if len(words) > 0:
                last_word = words[-1]
                # Replace formal ending with informal
                for formal in self.formal_verb_endings:
                    if formal in last_word:
                        words[-1] = last_word.replace(formal, random.choice(self.informal_verb_endings))
                        return ' '.join(words), "SEM:HONOR_CTX"
        
        elif has_informal:
            # Add formal verb ending  
            if len(words) > 0:
                last_word = words[-1]
                for informal in self.informal_verb_endings:
                    if informal in last_word:
                        words[-1] = last_word.replace(informal, random.choice(self.formal_verb_endings))
                        return ' '.join(words), "SEM:HONOR_CTX"
        
        return sentence, "NONE"
    
    def create_argument_reversal(self, sentence: str) -> Tuple[str, str]:
        """
        Create error: Reverse agent and patient
        Example: "दूधले गाई दिन्छ" (Milk gives cow) instead of "गाईले दूध दिन्छ" (Cow gives milk)
        """
        words = sentence.split()
        
        if len(words) >= 3:
            # Simple reversal of first two nouns
            temp = words[0]
            words[0] = words[1]
            words[1] = temp
            return ' '.join(words), "SEM:ARG_REVERSAL"
        
        return sentence, "NONE"
    
    def create_semantic_anomaly(self, sentence: str) -> Tuple[str, str]:
        """
        Create general semantic implausibility while maintaining grammar
        Uses multiple strategies
        """
        strategies = [
            self.create_animate_inanimate_violation,
            self.create_object_type_violation,
            self.create_honorific_context_violation,
            self.create_argument_reversal,
        ]
        
        # Try each strategy
        for strategy in strategies:
            result, error_type = strategy(sentence)
            if error_type != "NONE":
                return result, error_type
        
        return sentence, "NONE"
    
    def generate_semantic_error_sentence(
        self, 
        clean_sentence: str, 
        error_type: str = None
    ) -> Tuple[str, str]:
        """
        Generate a semantic error in a clean sentence
        
        Args:
            clean_sentence: Grammatically correct Nepali sentence
            error_type: Specific error type or None for random
        
        Returns:
            (semantically_anomalous_sentence, error_type_code)
        """
        if error_type == "SEM:ANIM_VIOLATION":
            return self.create_animate_inanimate_violation(clean_sentence)
        elif error_type == "SEM:OBJ_TYPE":
            return self.create_object_type_violation(clean_sentence)
        elif error_type == "SEM:HONOR_CTX":
            return self.create_honorific_context_violation(clean_sentence)
        elif error_type == "SEM:ARG_REVERSAL":
            return self.create_argument_reversal(clean_sentence)
        else:
            return self.create_semantic_anomaly(clean_sentence)
    
    def generate_complex_example(self, base_sentence: str) -> Dict:
        """
        Generate a complex example with multiple error types
        Useful for creating challenging test cases
        """
        from rule_based_errors import NepaliErrorGenerator
        
        # Start with semantic error
        semantic_error, sem_type = self.generate_semantic_error_sentence(base_sentence)
        
        # Add grammatical error on top
        gram_generator = NepaliErrorGenerator()
        final_error, gram_type = gram_generator.generate_error(semantic_error)
        
        return {
            'original': base_sentence,
            'semantic_error': semantic_error,
            'final_error': final_error,
            'error_types': [sem_type, gram_type],
            'is_multi_error': True
        }


# Example usage
if __name__ == "__main__":
    sem_gen = SemanticErrorGenerator()
    
    # Test sentences with expected semantic structure
    test_cases = [
        "बच्चाले खाना खान्छ",  # Child eats food
        "शिक्षकले विद्यार्थीलाई पढाउँछन्",  # Teacher teaches student
        "गाईले दूध दिन्छ",  # Cow gives milk
        "म घरमा बस्छु",  # I live at home
        "तपाईं कहाँ जानुहुन्छ",  # Where are you going (formal)
    ]
    
    print("=== Semantic Error Generation Examples ===\n")
    
    for sent in test_cases:
        error_sent, error_type = sem_gen.generate_semantic_error_sentence(sent)
        print(f"Clean:    {sent}")
        print(f"Semantic: {error_sent}")
        print(f"Type:     {error_type}")
        print()
    
    print("\n=== Complex Multi-Error Examples ===\n")
    for sent in test_cases[:2]:
        result = sem_gen.generate_complex_example(sent)
        print(f"Original:       {result['original']}")
        print(f"Semantic Error: {result['semantic_error']}")
        print(f"Final Error:    {result['final_error']}")
        print(f"Error Types:    {', '.join(result['error_types'])}")
        print()