"""
Advanced Semantic Error Generation for Nepali GEC
Based on linguistic theory: selectional restrictions, semantic roles, and world knowledge

This generator creates DIVERSE semantic errors that are grammatically correct
but semantically implausible, covering:
1. Selectional preference violations
2. Thematic role reversals
3. Semantic type mismatches
4. World knowledge violations
5. Metaphysical impossibilities
6. Pragmatic anomalies
"""
import random
import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SemanticFrame:
    """Represents a verb's semantic frame with role constraints"""
    verb: str
    required_roles: Dict[str, List[str]]  # role -> allowed semantic types
    optional_roles: Dict[str, List[str]]
    
    
class NepaliSemanticKnowledgeBase:
    """
    Comprehensive semantic knowledge base for Nepali
    Built on linguistic principles, not just examples
    """
    
    def __init__(self):
        self._initialize_semantic_types()
        self._initialize_verb_frames()
        self._initialize_world_knowledge()
        self._initialize_incompatibilities()
    
    def _initialize_semantic_types(self):
        """Define hierarchical semantic types"""
        
        # Animate entities
        self.HUMAN = {
            'मान्छे', 'मानिस', 'व्यक्ति', 'केटा', 'केटी', 'बच्चा', 'बालक', 'बालिका',
            'शिक्षक', 'शिक्षिका', 'विद्यार्थी', 'डाक्टर', 'नर्स', 'किसान',
            'आमा', 'बुबा', 'दाजु', 'दिदी', 'भाइ', 'बहिनी', 'साथी', 'छोरा', 'छोरी',
            'महिला', 'पुरुष', 'युवक', 'युवती', 'वृद्ध', 'नागरिक', 'मजदुर', 'व्यापारी'
        }
        
        self.ANIMAL = {
            'कुकुर', 'बिरालो', 'गाई', 'भैंसी', 'बाख्रा', 'भेडा', 'सुँगुर',
            'चरा', 'काग', 'परेवा', 'कौवा', 'हात्ती', 'घोडा', 'गधा',
            'माछा', 'सर्प', 'साँप', 'चील', 'गिद्ध', 'बाघ', 'सिंह', 'हिरण'
        }
        
        self.PLANT = {
            'रुख', 'बोट', 'फूल', 'गुलाफ', 'कमल', 'घाँस', 'बाँस',
            'आँप', 'केरा', 'सुन्तला', 'अमरुद', 'धान', 'गहुँ', 'मकै'
        }
        
        # Inanimate entities
        self.ARTIFACT = {
            'किताब', 'कलम', 'कागज', 'झोला', 'थैला', 'घडी', 'मोबाइल', 'कम्प्युटर',
            'कुर्सी', 'टेबल', 'झ्याल', 'ढोका', 'बत्ती', 'पंखा', 'तस्बिर',
            'गाडी', 'साइकल', 'मोटरसाइकल', 'बस', 'हवाईजहाज', 'जुत्ता', 'लुगा'
        }
        
        self.NATURAL_OBJECT = {
            'ढुङ्गा', 'चट्टान', 'पहाड', 'खोला', 'नदी', 'समुद्र', 'तालाब',
            'आकाश', 'बादल', 'पानी', 'हावा', 'आगो', 'धुलो', 'माटो'
        }
        
        self.FOOD = {
            'खाना', 'भात', 'दाल', 'तरकारी', 'रोटी', 'चपाती', 'परौठा',
            'दूध', 'दही', 'मासु', 'माछा', 'अण्डा', 'फलफूल', 'केरा',
            'आँप', 'स्याउ', 'चिया', 'कफी', 'पानी', 'जुस', 'मिठाई'
        }
        
        self.LIQUID = {'पानी', 'दूध', 'दही', 'चिया', 'कफी', 'जुस', 'तेल', 'मह'}
        
        self.ABSTRACT = {
            'विचार', 'सपना', 'आशा', 'डर', 'खुशी', 'दुःख', 'क्रोध', 'माया', 'घृणा',
            'समय', 'स्वतन्त्रता', 'शान्ति', 'युद्ध', 'प्रेम', 'सत्य', 'न्याय',
            'ज्ञान', 'शक्ति', 'साहस', 'धैर्य', 'आत्मा', 'मन', 'हृदय'
        }
        
        self.PLACE = {
            'घर', 'विद्यालय', 'महाविद्यालय', 'विश्वविद्यालय', 'अस्पताल',
            'बजार', 'पसल', 'होटल', 'रेष्टुरेन्ट', 'कार्यालय', 'कारखाना',
            'मन्दिर', 'मस्जिद', 'गुम्बा', 'गिर्जाघर', 'पार्क', 'मैदान',
            'गाउँ', 'शहर', 'नगर', 'देश', 'राष्ट्र', 'संसार'
        }
        
        # Composite categories
        self.ANIMATE = self.HUMAN | self.ANIMAL
        self.LIVING = self.ANIMATE | self.PLANT
        self.PHYSICAL_OBJECT = self.ARTIFACT | self.NATURAL_OBJECT | self.LIVING
        self.CONCRETE = self.PHYSICAL_OBJECT | self.FOOD | self.LIQUID
        self.INANIMATE = self.ARTIFACT | self.NATURAL_OBJECT | self.FOOD | self.LIQUID
        
        # Edibles
        self.EDIBLE = self.FOOD | {'पानी', 'दूध', 'दही', 'चिया', 'फलफूल'}
        self.INEDIBLE = self.ARTIFACT | self.NATURAL_OBJECT | self.ABSTRACT | self.PLACE
        
        # Create reverse mapping: entity -> types
        self.entity_types = defaultdict(set)
        for entity in self.HUMAN:
            self.entity_types[entity].update(['HUMAN', 'ANIMATE', 'LIVING', 'CONCRETE'])
        for entity in self.ANIMAL:
            self.entity_types[entity].update(['ANIMAL', 'ANIMATE', 'LIVING', 'CONCRETE'])
        for entity in self.PLANT:
            self.entity_types[entity].update(['PLANT', 'LIVING', 'CONCRETE'])
        for entity in self.ARTIFACT:
            self.entity_types[entity].update(['ARTIFACT', 'INANIMATE', 'CONCRETE'])
        for entity in self.ABSTRACT:
            self.entity_types[entity].add('ABSTRACT')
        for entity in self.PLACE:
            self.entity_types[entity].update(['PLACE', 'INANIMATE'])
        for entity in self.EDIBLE:
            self.entity_types[entity].add('EDIBLE')
    
    def _initialize_verb_frames(self):
        """
        Define selectional restrictions for verbs
        Based on semantic role theory (Agent, Patient, Theme, Experiencer, etc.)
        """
        
        self.verb_frames = {
            # Agentive verbs - require animate agent
            'खान्छ': {
                'agent': ['ANIMATE'],  # Who eats
                'theme': ['EDIBLE'],  # What is eaten
                'incompatible_agent': ['ABSTRACT', 'PLACE'],
                'incompatible_theme': ['ABSTRACT', 'ANIMATE']
            },
            'खान्छन्': {
                'agent': ['ANIMATE'],
                'theme': ['EDIBLE'],
                'incompatible_agent': ['ABSTRACT', 'PLACE'],
                'incompatible_theme': ['ABSTRACT', 'ANIMATE']
            },
            'पिउँछ': {
                'agent': ['ANIMATE'],
                'theme': ['LIQUID'],
                'incompatible_agent': ['INANIMATE'],
                'incompatible_theme': ['CONCRETE', 'ABSTRACT']
            },
            
            # Cognitive verbs - require human/high-level animate
            'सोच्छ': {
                'agent': ['HUMAN'],
                'theme': ['ABSTRACT', 'ANY'],
                'incompatible_agent': ['INANIMATE', 'PLANT']
            },
            'सिक्छ': {
                'agent': ['HUMAN'],
                'theme': ['ABSTRACT', 'ANY'],
                'incompatible_agent': ['INANIMATE', 'ANIMAL']
            },
            'पढ्छ': {
                'agent': ['HUMAN'],
                'theme': ['ARTIFACT', 'ABSTRACT'],
                'incompatible_agent': ['INANIMATE'],
                'incompatible_theme': ['ANIMATE', 'LIQUID']
            },
            'लेख्छ': {
                'agent': ['HUMAN'],
                'theme': ['ABSTRACT', 'ARTIFACT'],
                'incompatible_agent': ['INANIMATE', 'ANIMAL']
            },
            
            # Communication verbs
            'बोल्छ': {
                'agent': ['HUMAN'],
                'incompatible_agent': ['INANIMATE', 'PLANT']
            },
            'भन्छ': {
                'agent': ['HUMAN'],
                'incompatible_agent': ['INANIMATE']
            },
            
            # Motion verbs - require entity capable of self-motion
            'हिँड्छ': {
                'agent': ['ANIMATE'],
                'incompatible_agent': ['INANIMATE', 'ABSTRACT', 'PLANT']
            },
            'दौडन्छ': {
                'agent': ['ANIMATE'],
                'incompatible_agent': ['INANIMATE', 'PLANT']
            },
            'उड्छ': {
                'agent': ['ANIMAL'],  # Birds, insects
                'incompatible_agent': ['HUMAN', 'INANIMATE', 'PLANT']
            },
            
            # Creation/Production verbs
            'दिन्छ': {
                'agent': ['ANIMATE', 'LIVING'],
                'theme': ['CONCRETE', 'ABSTRACT'],
                'special': 'production'  # गाई दूध दिन्छ OK, दूध गाई दिन्छ NOT OK
            },
            'बनाउँछ': {
                'agent': ['HUMAN'],
                'theme': ['ARTIFACT', 'FOOD', 'ABSTRACT'],
                'incompatible_agent': ['INANIMATE']
            },
            
            # Perception verbs
            'देख्छ': {
                'agent': ['ANIMATE'],
                'theme': ['CONCRETE', 'ABSTRACT'],
                'incompatible_agent': ['INANIMATE', 'PLANT']
            },
            'सुन्छ': {
                'agent': ['ANIMATE'],
                'incompatible_agent': ['INANIMATE', 'PLANT']
            },
            
            # Emotion verbs - require experiencer
            'रुन्छ': {
                'experiencer': ['HUMAN'],
                'incompatible_experiencer': ['INANIMATE', 'ANIMAL']
            },
            'हाँस्छ': {
                'experiencer': ['HUMAN'],
                'incompatible_experiencer': ['INANIMATE']
            },
            
            # Teaching verbs
            'सिकाउँछ': {
                'agent': ['HUMAN'],
                'theme': ['ABSTRACT', 'ARTIFACT'],
                'recipient': ['HUMAN'],
                'incompatible_agent': ['INANIMATE', 'ANIMAL']
            },
            'पढाउँछ': {
                'agent': ['HUMAN'],
                'theme': ['ABSTRACT'],
                'recipient': ['HUMAN'],
                'incompatible_agent': ['INANIMATE']
            }
        }
    
    def _initialize_world_knowledge(self):
        """
        World knowledge constraints
        Things that are true/false about the world
        """
        
        self.world_facts = {
            # Production relations (X produces Y)
            'produces': {
                'गाई': ['दूध'],
                'भैंसी': ['दूध'],
                'बाख्रा': ['दूध'],
                'चरा': ['अण्डा'],
                'रुख': ['फलफूल', 'छाया'],
                'बोट': ['फूल']
            },
            
            # Consumption relations (X eats/drinks Y)
            'consumes': {
                'मान्छे': self.FOOD | self.LIQUID,
                'गाई': {'घाँस', 'दाना'},
                'कुकुर': {'मासु', 'भात', 'खाना'},
                'बिरालो': {'दूध', 'माछा', 'मासु'},
                'चरा': {'दाना', 'कीरा'}
            },
            
            # Location constraints (X lives in/at Y)
            'habitat': {
                'माछा': {'पानी', 'नदी', 'तालाब'},
                'चरा': {'रुख', 'आकाश'},
                'मान्छे': {'घर', 'गाउँ', 'शहर'}
            },
            
            # Size relations (X is bigger than Y)
            'larger_than': {
                'हात्ती': self.ANIMAL - {'हात्ती'},
                'घर': {'कुर्सी', 'टेबल', 'किताब'},
                'पहाड': {'रुख', 'घर', 'गाउँ'}
            }
        }
    
    def _initialize_incompatibilities(self):
        """Define semantic incompatibilities"""
        
        self.incompatible_pairs = [
            # Abstract cannot do physical actions
            ('विचार', 'खान्छ'),
            ('सपना', 'हिँड्छ'),
            ('समय', 'दौडन्छ'),
            
            # Inanimate cannot have agency
            ('किताब', 'सोच्छ'),
            ('ढुङ्गा', 'बोल्छ'),
            ('घर', 'रुन्छ'),
            
            # Type mismatches
            ('पानी', 'पढ्छ'),
            ('हावा', 'लेख्छ'),
        ]


class AdvancedSemanticErrorGenerator:
    """
    Generate diverse semantic errors using linguistic knowledge
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.kb = NepaliSemanticKnowledgeBase()
        
    def get_semantic_type(self, entity: str) -> Set[str]:
        """Get semantic types of an entity"""
        return self.kb.entity_types.get(entity, {'UNKNOWN'})
    
    def is_compatible(self, entity: str, required_types: List[str]) -> bool:
        """Check if entity satisfies type requirements"""
        entity_types = self.get_semantic_type(entity)
        return any(req_type in entity_types for req_type in required_types)
    
    def get_incompatible_entity(
        self,
        required_types: List[str],
        avoid_types: List[str] = None
    ) -> str:
        """Get an entity that violates type requirements"""
        avoid_types = avoid_types or []
        
        # Collect entities that DON'T match requirements
        incompatible = set()
        
        for entity, types in self.kb.entity_types.items():
            # Must NOT satisfy required types
            if not any(req in types for req in required_types):
                # And should match avoid types if specified
                if not avoid_types or any(avoid in types for avoid in avoid_types):
                    incompatible.add(entity)
        
        if incompatible:
            return random.choice(list(incompatible))
        
        # Fallback
        return random.choice(list(self.kb.ABSTRACT))
    
    def generate_selectional_violation(self, sentence: str) -> Tuple[str, str]:
        """
        Violate verb's selectional restrictions
        Example: किताब सोच्छ (book thinks) - inanimate with cognitive verb
        """
        words = sentence.split()
        
        # Find verb
        for word in words:
            for verb, frame in self.kb.verb_frames.items():
                if verb in word:
                    # Get agent requirements
                    if 'agent' in frame and len(words) > 0:
                        required = frame['agent']
                        incompatible = frame.get('incompatible_agent', [])
                        
                        # Replace agent with incompatible entity
                        new_agent = self.get_incompatible_entity(required, incompatible)
                        words[0] = new_agent
                        
                        return ' '.join(words), f"SEM:SELECT_VIOLATION:{verb}"
        
        return sentence, "NONE"
    
    def generate_object_type_mismatch(self, sentence: str) -> Tuple[str, str]:
        """
        Wrong object type for verb
        Example: म घर खान्छु (I eat house)
        """
        words = sentence.split()
        
        for i, word in enumerate(words):
            for verb, frame in self.kb.verb_frames.items():
                if verb in word and 'theme' in frame:
                    # Find object position (usually before verb in Nepali SOV)
                    if i > 0:
                        required = frame['theme']
                        incompatible = frame.get('incompatible_theme', [])
                        
                        new_object = self.get_incompatible_entity(required, incompatible)
                        words[i-1] = new_object
                        
                        return ' '.join(words), f"SEM:OBJ_TYPE:{verb}"
        
        return sentence, "NONE"
    
    def generate_world_knowledge_violation(self, sentence: str) -> Tuple[str, str]:
        """
        Violate world knowledge
        Example: दूधले गाई दिन्छ (milk gives cow) - reversal of production
        """
        words = sentence.split()
        
        # Look for production verb
        if 'दिन्छ' in sentence or 'दिन्छन्' in sentence:
            # Check for producer-product pair
            for producer, products in self.kb.world_facts['produces'].items():
                if producer in sentence:
                    # Reverse: make product the agent
                    if len(words) >= 2:
                        product = random.choice(list(products))
                        # Swap positions
                        temp = words[0]
                        words[0] = product
                        if len(words) > 1:
                            words[1] = temp
                        
                        return ' '.join(words), "SEM:WORLD_KNOWLEDGE:REVERSAL"
        
        return sentence, "NONE"
    
    def generate_consumption_violation(self, sentence: str) -> Tuple[str, str]:
        """
        Wrong food for entity
        Example: गाईले मासु खान्छ (cow eats meat) - herbivore eating meat
        """
        words = sentence.split()
        
        if any(v in sentence for v in ['खान्छ', 'खान्छन्', 'खान्छु']):
            # Find consumer
            for consumer, allowed_foods in self.kb.world_facts['consumes'].items():
                if consumer in words:
                    # Give wrong food
                    all_foods = self.kb.EDIBLE
                    wrong_foods = all_foods - allowed_foods
                    
                    if wrong_foods and len(words) > 1:
                        wrong_food = random.choice(list(wrong_foods))
                        # Replace object
                        for i in range(len(words) - 1):
                            if words[i] in self.kb.EDIBLE:
                                words[i] = wrong_food
                                return ' '.join(words), f"SEM:CONSUMPTION:{consumer}"
        
        return sentence, "NONE"
    
    def generate_metaphysical_impossibility(self, sentence: str) -> Tuple[str, str]:
        """
        Create metaphysically impossible situations
        Example: समयले खाना खान्छ (time eats food) - abstract doing physical action
        """
        words = sentence.split()
        
        # Replace agent with abstract concept for physical verb
        physical_verbs = {'खान्छ', 'हिँड्छ', 'दौडन्छ', 'लेख्छ', 'पढ्छ'}
        
        for verb in physical_verbs:
            if verb in sentence and len(words) > 0:
                abstract = random.choice(list(self.kb.ABSTRACT))
                words[0] = abstract
                return ' '.join(words), f"SEM:METAPHYSICAL:{verb}"
        
        return sentence, "NONE"
    
    def generate_category_error(self, sentence: str) -> Tuple[str, str]:
        """
        Category mistakes (treating one type as another)
        Example: स्वतन्त्रता दौडन्छ (freedom runs)
        """
        words = sentence.split()
        
        # Motion verbs with abstract subjects
        motion_verbs = {'हिँड्छ', 'दौडन्छ', 'उड्छ', 'जान्छ'}
        
        for verb in motion_verbs:
            if verb in sentence:
                # Replace with abstract/place
                wrong_agent = random.choice(list(self.kb.ABSTRACT | self.kb.PLACE))
                if len(words) > 0:
                    words[0] = wrong_agent
                    return ' '.join(words), f"SEM:CATEGORY_ERROR:{verb}"
        
        return sentence, "NONE"
    
    def generate_semantic_error(
        self,
        sentence: str,
        error_type: str = None
    ) -> Tuple[str, str]:
        """
        Generate any semantic error
        
        Strategies:
        1. Selectional restriction violation
        2. Object type mismatch
        3. World knowledge violation
        4. Consumption violation
        5. Metaphysical impossibility
        6. Category error
        """
        strategies = [
            self.generate_selectional_violation,
            self.generate_object_type_mismatch,
            self.generate_world_knowledge_violation,
            self.generate_consumption_violation,
            self.generate_metaphysical_impossibility,
            self.generate_category_error
        ]
        
        # Try each strategy
        random.shuffle(strategies)
        
        for strategy in strategies:
            result, etype = strategy(sentence)
            if etype != "NONE":
                return result, etype
        
        return sentence, "NONE"


# Testing and examples
if __name__ == "__main__":
    gen = AdvancedSemanticErrorGenerator()
    
    test_sentences = [
        "बच्चाले खाना खान्छ",
        "शिक्षकले विद्यार्थीलाई पढाउँछन्",
        "गाईले दूध दिन्छ",
        "म किताब पढ्छु",
        "केटाले पानी पिउँछ",
        "मान्छेले घर बनाउँछ",
        "चराले उड्छ",
        "विद्यार्थीले सिक्छ"
    ]
    
    print("=" * 80)
    print("ADVANCED SEMANTIC ERROR GENERATION")
    print("Demonstrating DIVERSE semantic violations")
    print("=" * 80)
    
    for sent in test_sentences:
        error, etype = gen.generate_semantic_error(sent)
        
        print(f"\nClean:    {sent}")
        print(f"Error:    {error}")
        print(f"Type:     {etype}")
        
        # Show what was violated
        if etype != "NONE":
            print(f"✓ Semantically implausible but grammatically correct!")