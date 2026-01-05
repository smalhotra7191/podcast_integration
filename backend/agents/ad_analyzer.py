"""
Ad Script Analyzer Agent
Analyzes ad scripts to extract product information and problem being solved
Uses HuggingFace open LLMs for analysis
"""
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re

class AdScriptAnalyzer:
    def __init__(self):
        """Initialize the ad script analyzer with HuggingFace models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "microsoft/DialoGPT-medium"  # Free, open model
        
        # For text generation/analysis, we'll use a smaller open model
        # that works well for extraction tasks
        self.summarizer = None
        self.qa_model = None
        
    def _load_models(self):
        """Lazy load models to save memory"""
        if self.summarizer is None:
            print("Loading summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
        
        if self.qa_model is None:
            print("Loading QA model...")
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if self.device == "cuda" else -1
            )
    
    def read_script(self, script_path: str) -> str:
        """Read the ad script from file"""
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def analyze(self, script_path: str) -> dict:
        """
        Analyze the ad script to extract:
        - Product name and description
        - Problem being solved (critical for placement)
        - Problem-related keywords for searching in podcast
        - Target audience
        - Key selling points
        - Call to action
        """
        script_content = self.read_script(script_path)
        
        # Load models
        self._load_models()
        
        analysis = {
            'product_name': self._extract_product_name(script_content),
            'product_description': self._extract_product_description(script_content),
            'problem_solved': self._extract_problem(script_content),
            'problem_keywords': self.get_problem_keywords(script_content),  # New: for searching in podcast
            'target_audience': self._extract_target_audience(script_content),
            'key_benefits': self._extract_benefits(script_content),
            'call_to_action': self._extract_cta(script_content),
            'keywords': self._extract_keywords(script_content),
            'tone': self._analyze_tone(script_content),
            'full_script': script_content
        }
        
        return analysis
    
    def _extract_product_name(self, text: str) -> str:
        """Extract the product/service name from the script"""
        # Try QA approach
        try:
            result = self.qa_model(
                question="What is the name of the product or service being advertised?",
                context=text
            )
            if result['score'] > 0.1:
                return result['answer']
        except:
            pass
        
        # Fallback: Look for common patterns
        patterns = [
            r'introducing\s+([A-Z][a-zA-Z\s]+)',
            r'welcome to\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z]+)\s+is\s+(?:the|a|an)',
            r'try\s+([A-Z][a-zA-Z\s]+)\s+today'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Product/Service"
    
    def _extract_product_description(self, text: str) -> str:
        """Extract product description"""
        try:
            # Use summarizer to get a concise description
            if len(text) > 100:
                summary = self.summarizer(
                    text, 
                    max_length=100, 
                    min_length=30, 
                    do_sample=False
                )
                return summary[0]['summary_text']
        except:
            pass
        
        # Return first 200 characters as fallback
        return text[:200] + "..." if len(text) > 200 else text
    
    def _extract_problem(self, text: str) -> str:
        """Extract the problem being solved - this is critical for ad placement"""
        problems_found = []
        
        # Try QA approach first
        try:
            result = self.qa_model(
                question="What problem does this product solve?",
                context=text
            )
            if result['score'] > 0.05:
                problems_found.append(result['answer'])
        except:
            pass
        
        # Look for problem-related patterns
        problem_patterns = [
            r'tired of\s+([^.?!]+)',
            r'struggling with\s+([^.?!]+)',
            r'problem(?:s)? (?:of|with)\s+([^.?!]+)',
            r'(?:do you|have you)\s+(?:ever\s+)?([^.?!]+\?)',
            r'(?:frustrated|annoyed|bothered) (?:by|with)\s+([^.?!]+)',
            r'(?:hard|difficult|challenging) to\s+([^.?!]+)',
            r'(?:hate|dislike|can\'t stand)\s+([^.?!]+)',
            r'(?:waste|wasting)\s+(?:time|money|hours)\s+(?:on|with)?\s*([^.?!]+)',
            r'(?:never|don\'t)\s+have\s+(?:enough\s+)?(?:time|energy)\s+(?:for|to)\s+([^.?!]+)',
            r'(?:overwhelmed|stressed|anxious)\s+(?:by|about|with)\s+([^.?!]+)',
            r'(?:worried|concerned)\s+about\s+([^.?!]+)',
            r'(?:can\'t|cannot|couldn\'t)\s+(?:seem to|figure out|manage)\s+([^.?!]+)',
        ]
        
        for pattern in problem_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip().rstrip('?.,!')
                if len(cleaned) > 10:  # Minimum length for meaningful problem
                    problems_found.append(cleaned)
        
        if problems_found:
            # Return the most detailed problem found
            return max(problems_found, key=len)
        
        return "General improvement/enhancement"
    
    def get_problem_keywords(self, text: str) -> list:
        """
        Extract key phrases that describe the problem being solved.
        These will be used to search for problem mentions in the podcast.
        """
        problem_keywords = []
        
        # Get the main problem description
        main_problem = self._extract_problem(text)
        if main_problem and main_problem != "General improvement/enhancement":
            # Extract key noun phrases from the problem
            words = main_problem.lower().split()
            # Create n-grams (2-3 word phrases)
            for i in range(len(words)):
                if i < len(words) - 1:
                    problem_keywords.append(' '.join(words[i:i+2]))
                if i < len(words) - 2:
                    problem_keywords.append(' '.join(words[i:i+3]))
            problem_keywords.append(main_problem.lower())
        
        # Also look for explicit problem indicators in the text
        text_lower = text.lower()
        
        # Pattern-based keyword extraction
        indicator_patterns = [
            r'(?:tired of|struggling with|problem with|hard to|difficult to)\s+(\w+(?:\s+\w+)?)',
            r'(?:save|saving)\s+(?:time|money|hours)\s+(?:on|with)?\s*(\w+(?:\s+\w+)?)',
            r'(?:manage|managing|track|tracking)\s+(\w+)',
            r'(?:improve|improving|better)\s+(\w+)',
        ]
        
        for pattern in indicator_patterns:
            matches = re.findall(pattern, text_lower)
            problem_keywords.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in problem_keywords:
            if kw not in seen and len(kw) > 3:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # Return top 10 problem-related keywords
    
    def _extract_target_audience(self, text: str) -> str:
        """Identify target audience"""
        try:
            result = self.qa_model(
                question="Who is the target audience for this product?",
                context=text
            )
            if result['score'] > 0.05:
                return result['answer']
        except:
            pass
        
        # Look for audience indicators
        audience_patterns = [
            r'for\s+([\w\s]+(?:professionals|entrepreneurs|businesses|students|parents|developers|creators))',
            r'(?:whether you\'re|if you\'re)\s+(?:a\s+)?([\w\s]+)',
            r'([\w\s]+)\s+(?:will|can|should)\s+(?:love|benefit|use)'
        ]
        
        for pattern in audience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "General audience"
    
    def _extract_benefits(self, text: str) -> list:
        """Extract key benefits/selling points"""
        benefits = []
        
        # Look for benefit patterns
        benefit_patterns = [
            r'(?:helps? you|allows? you to|enables? you to)\s+([^.!?]+)',
            r'(?:save|increase|improve|boost|enhance)\s+([^.!?]+)',
            r'(?:get|achieve|experience)\s+([^.!?]+)',
            r'(?:no more|never again)\s+([^.!?]+)'
        ]
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            benefits.extend([m.strip() for m in matches[:3]])
        
        return benefits[:5] if benefits else ["Quality product", "Great value"]
    
    def _extract_cta(self, text: str) -> str:
        """Extract call to action"""
        cta_patterns = [
            r'(visit\s+[\w.]+)',
            r'(go to\s+[\w.]+)',
            r'(use (?:code|coupon)\s+[\w]+)',
            r'(sign up\s+[^.!?]+)',
            r'(download\s+[^.!?]+)',
            r'(try\s+[^.!?]+(?:free|today))',
            r'(get\s+(?:started|your)[^.!?]+)'
        ]
        
        for pattern in cta_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Visit our website to learn more"
    
    def _extract_keywords(self, text: str) -> list:
        """Extract important keywords for matching with podcast content"""
        # Remove common words and extract meaningful terms
        text_lower = text.lower()
        
        # Common stop words to ignore
        stop_words = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'up', 'about', 'into', 'over', 'after', 'and', 'but', 'or',
            'as', 'if', 'when', 'than', 'because', 'while', 'although', 'though',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'how', 'why',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'just', 'now',
            'very', 'your', 'our', 'their', 'my', 'his', 'her', 'its'
        ])
        
        # Extract words
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        
        # Count frequency and filter
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:15]]
    
    def _analyze_tone(self, text: str) -> str:
        """Analyze the tone of the ad"""
        text_lower = text.lower()
        
        # Simple tone detection based on keywords
        if any(word in text_lower for word in ['exciting', 'amazing', 'incredible', 'awesome', 'revolutionary']):
            return "enthusiastic"
        elif any(word in text_lower for word in ['professional', 'expert', 'enterprise', 'business']):
            return "professional"
        elif any(word in text_lower for word in ['fun', 'enjoy', 'love', 'happy', 'friend']):
            return "friendly"
        elif any(word in text_lower for word in ['trust', 'reliable', 'secure', 'proven', 'guaranteed']):
            return "trustworthy"
        else:
            return "conversational"


# Test function
if __name__ == "__main__":
    analyzer = AdScriptAnalyzer()
    
    # Test with sample script
    test_script = """
    Are you tired of spending hours trying to manage your finances? 
    Introducing BudgetPro - the revolutionary app that helps you take control of your money.
    
    Whether you're a busy professional or a student on a tight budget, BudgetPro makes 
    financial management simple. Track expenses, set savings goals, and get personalized 
    insights - all in one place.
    
    With BudgetPro, you'll never miss a bill again. Our smart notifications keep you 
    on track, and our AI-powered suggestions help you save more every month.
    
    Download BudgetPro today and use code SAVE20 for 20% off your first year!
    Visit budgetpro.com to get started.
    """
    
    # Write test script to file
    with open("test_ad.txt", "w") as f:
        f.write(test_script)
    
    result = analyzer.analyze("test_ad.txt")
    print("Analysis Result:")
    for key, value in result.items():
        if key != 'full_script':
            print(f"  {key}: {value}")
    
    os.remove("test_ad.txt")
