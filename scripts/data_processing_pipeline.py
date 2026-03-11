import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from utils.markdown_to_json import (
    MarkdownToJsonConverter
)

# LLM-related imports
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("OpenAI library not available. LLM features will be disabled.")


# ==================== Data Structure Definitions ====================

@dataclass
class OutlineItem:
    """
    Outline item data structure.

    Attributes:
        level: Heading level (1, 2, 3...)
        title: Heading text
    """
    level: int
    title: str
    
    def to_list(self) -> List[Union[int, str]]:
        """Convert to [level, title] format."""
        return [self.level, self.title]
    
    @classmethod
    def from_list(cls, item: List) -> 'OutlineItem':
        """Create from [level, title] format."""
        if len(item) != 2:
            raise ValueError(f"Invalid outline item format: {item}")
        return cls(level=item[0], title=item[1])

@dataclass
class Outline:
    """
    Outline data structure.

    Attributes:
        items: List of outline items
    """
    items: List[OutlineItem]

    def to_list(self) -> List[List]:
        """Convert to [level, title] format."""
        return [item.to_list() for item in self.items]
    
    @classmethod
    def from_list(cls, items: List[List]) -> 'Outline':
        """Create from [level, title] format."""
        return cls(items=[OutlineItem.from_list(item) for item in items])

@dataclass
class ContentSection:
    """
    Content section data structure.

    Attributes:
        heading: Section heading
        level: Section level
        content: Section content text
        stats: Statistics (optional)
    """
    heading: str
    level: int
    content: str
    stats: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format."""
        result = {
            'heading': self.heading,
            'level': self.level,
            'content': self.content
        }
        if self.stats:
            result['stats'] = self.stats
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContentSection':
        """Create from dict format."""
        return cls(
            heading=data.get('heading', ''),
            level=data.get('level', 0),
            content=data.get('content', ''),
            stats=data.get('stats')
        )

@dataclass
class Content:
    """
    Content data structure.

    Attributes:
        sections: List of content sections
    """
    sections: List[ContentSection]

    def to_list(self) -> List[Dict]:
        """Convert to dict format."""
        return [section.to_dict() for section in self.sections]
    
    @classmethod
    def from_list(cls, sections: List[Dict]) -> 'Content':
        """Create from dict format."""
        return cls(sections=[ContentSection.from_dict(section) for section in sections])

@dataclass
class ReferenceEntry:
    """
    Reference entry data structure.

    Attributes:
        text: Reference text
        number: Reference number
        title: Reference title
    """
    text: str
    number: int | None
    title: str
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format."""
        return {
            'text': self.text,
            'number': self.number,
            'title': self.title
        }
    @classmethod
    def from_dict(cls, data: Union[Dict, str]) -> 'ReferenceEntry':
        """Create from dict or string format."""
        if isinstance(data, str):
            # If string, parse number and title
            return cls(text=data, number=None, title=data)
        else:
            # If dict, extract from dict
            return cls(
                text=data.get('text', ''),
                number=data.get('number'),
                title=data.get('title', '')
            )

@dataclass
class References:
    """
    References data structure.

    Attributes:
        entries: List of reference entries
    """
    entries: List[ReferenceEntry]

    def to_list(self) -> List[Dict]:
        """Convert to dict format."""
        return [entry.to_dict() for entry in self.entries]
    
    @classmethod
    def from_list(cls, entries: List[Union[Dict, str]]) -> 'References':
        """Create from a list of dicts or strings."""
        return cls(entries=[ReferenceEntry.from_dict(entry) for entry in entries])

@dataclass
class SurveyData:
    """
    Complete survey data structure.

    Attributes:
        outline: Outline list
        content: Content sections list
        references: References list
        metadata: Metadata (optional)
    """
    outline: Outline = field(default_factory=Outline)
    content: Content = field(default_factory=Content)
    references: References = field(default_factory=References)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format (for JSON serialization)."""
        result = {
            'outline': self.outline.to_list(),
            'content': self.content.to_list(),
            'references': self.references.to_list()
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SurveyData':
        """Create from dict format."""
        return cls(
            outline=Outline.from_list(data.get('outline', [])),
            content=Content.from_list(data.get('content', [])),
            references=References.from_list(data.get('references', [])),
            metadata=data.get('metadata')
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data statistics."""
        total_content = ' '.join([section.content for section in self.content.sections])
        return {
            'outline_count': len(self.outline.items),
            'section_count': len(self.content.sections),
            'reference_count': len(self.references.entries),
            'total_chars': len(total_content),
            'total_words': len(total_content.split()),
            'sections_with_content': sum(1 for s in self.content.sections if s.content.strip())
        }


# ==================== Configuration ====================

@dataclass
class DataProcessingConfig:
    """Data processing configuration."""
    # Path configuration
    input_dir: str = "results/original"  # Input directory
    output_dir: str = "results/processed"  # Output directory
    systems: Optional[List[str]] = None  # Systems to process (e.g. Autosurvey), None means all
    categories: Optional[List[str]] = None  # Categories to process (e.g. Biology, Computer Science), None means all
    auto_convert_markdown: bool = True  # Whether to auto-convert Markdown files to JSON
    overwrite_original_json: bool = False  # Whether to overwrite existing original JSON files when converting from Markdown

    # Outline configuration
    normalize_outline: bool = True  # Whether to normalize outline
    llm_calibration: bool = False  # Whether to use LLM for outline calibration
    remove_empty_titles: bool = True  # Whether to remove empty titles

    # Content configuration
    normalize_content: bool = True  # Whether to normalize content
    remove_short_sections: bool = False  # Whether to remove short sections
    min_section_words: int = 10  # Minimum section word count
    calculate_section_stats: bool = True  # Whether to calculate section statistics

    # Reference configuration
    normalize_references: bool = True  # Whether to normalize references
    keep_number: bool = False  # Whether to keep reference numbers
    llm_quality_check: bool = False  # Whether to use LLM for quality check
    remove_duplicate_refs: bool = True  # Whether to remove duplicate references

    # LLM configuration
    llm_api_base: str = None  # OpenRouter API base URL
    llm_api_key: Optional[str] = None  # LLM API Key (read from environment variable)
    llm_model: str = None  # LLM model name
    llm_temperature: float = 0.0  # LLM temperature parameter
    llm_enable_reasoning: bool = False  # Whether to enable reasoning

    # Quality check configuration
    enable_quality_check: bool = False  # Whether to enable quality check
    min_outline_items: int = 3  # Minimum outline items
    min_references: int = 5  # Minimum references
    min_total_content_length: int = 1000  # Minimum total content length (characters)

    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "data_processing.log"
    
    def get_systems(self, base_dir: Optional[str] = None) -> List[str]:
        """
        Get list of systems to process.

        Args:
            base_dir: Base directory, defaults to input_dir

        Returns:
            List[str]: System list (e.g. Autosurvey)
        """
        if self.systems:
            return self.systems
        
        base_dir = base_dir or self.input_dir
        if not os.path.exists(base_dir):
            return []
        
        # Return all subdirectories as systems
        return [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    def get_categories_in_system(self, system: str, base_dir: Optional[str] = None) -> List[str]:
        """
        Get list of categories under a specific system.

        Args:
            system: System name (e.g. Autosurvey)
            base_dir: Base directory, defaults to input_dir

        Returns:
            List[str]: Category list (e.g. Biology, Computer Science)
        """
        base_dir = base_dir or self.input_dir
        system_path = os.path.join(base_dir, system)
        
        if not os.path.exists(system_path):
            return []
        
        if self.categories:
            # Filter specified categories
            available = [d for d in os.listdir(system_path) 
                        if os.path.isdir(os.path.join(system_path, d))]
            return [c for c in self.categories if c in available]
        
        # Return all categories
        return [d for d in os.listdir(system_path) 
                if os.path.isdir(os.path.join(system_path, d))]


# ==================== Helper Functions (Stateless) ====================

def setup_logging(config: DataProcessingConfig) -> logging.Logger:
    """
    Configure the logging system.

    Args:
        config: Data processing configuration

    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text: remove extra whitespace, unify line breaks, etc.

    Args:
        text: Raw text

    Returns:
        str: Normalized text
    """
    # Unify line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove trailing whitespace from lines
    text = re.sub(r'[ \t]+\n', '\n', text)

    # Merge multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def strip_leading_non_letters(text: str) -> str:
    """
    Remove all content before the first letter (e.g. numbers, symbols).
    """
    if not text:
        return text
    # Keep content starting from the first letter; return original if no letters found
    stripped = re.sub(r'^[^A-Za-z]+', '', text)
    return stripped if stripped else text


def remove_noise_patterns(text: str) -> str:
    """
    Remove common noise patterns.

    Args:
        text: Raw text

    Returns:
        str: Cleaned text
    """
    # Remove noise from image paths
    text = re.sub(r'!\[.*?\]\(images/[a-f0-9]+\.jpg\)', '[Image]', text)

    # Remove overly long math formulas (possibly OCR errors)
    text = re.sub(r'\$\$[^\$]{500,}\$\$', '[Formula]', text)

    # Remove repeated special characters
    text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)

    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'ftp://[^\s]+', '', text)
    text = re.sub(r'mailto://[^\s]+', '', text)
    text = re.sub(r'tel://[^\s]+', '', text)
    text = re.sub(r'file://[^\s]+', '', text)
    text = re.sub(r'magnet://[^\s]+', '', text)
    text = re.sub(r'irc://[^\s]+', '', text)
    text = re.sub(r'ircs://[^\s]+', '', text)
    return text


def clean_reference_text(ref: str) -> str:
    """
    Clean a single reference text (simple version for list cleanup).

    Args:
        ref: Raw reference text

    Returns:
        str: Cleaned reference
    """
    # Normalize whitespace
    ref = ' '.join(ref.split())

    # Strip leading/trailing whitespace
    ref = ref.strip()

    return ref


def extract_reference_title(reference: str, is_qwen_format: bool = False, is_gemini_format: bool = False) -> Tuple[Optional[int], str]:
    """
    Extract number and title from reference text.

    Pattern categories:
    - Pattern A (1, 2, 4, 5, 6, 8): Standard format [num] or num. prefix, extract title directly
    - Pattern B (3): [cite: num] format, contains journal info, needs cleanup
    - Pattern C (7, 9): Title followed by author name list, needs separation
    - Pattern D (qwen): Author, et al. (year). Title. *Journal*, vol(issue), pages
    - Pattern E (gemini): num. **Author** (year). "Title." *Journal*, vol(issue), pages

    Args:
        reference: Raw reference text
        is_qwen_format: Whether the reference is in qwen-generated format
        is_gemini_format: Whether the reference is in gemini-generated format

    Returns:
        Tuple[Optional[int], str]: (number, title)
    """
    if not reference or not reference.strip():
        return (None, "")

    # ===== Preprocessing =====
    # Remove line breaks
    text = reference.replace('\n', ' ').replace('\r', ' ')

    # Save original text for format detection
    original_text = text

    # Convert to lowercase (for processing, but some formats use original case)
    text_lower = text.lower()

    # Remove extra spaces
    text_lower = ' '.join(text_lower.split())

    if not text_lower:
        return (None, "")

    # ===== Detect gemini format =====
    # Gemini format: num. **Author** (year). "Title." *Journal*
    # Title is within quotes
    if is_gemini_format or re.search(r'\d+\.\s*\*\*.*?\*\*\s*\(\d{4}\)\s*\.\s*"[^"]+"', original_text):
        # Use original case text for extraction
        text = ' '.join(original_text.split())

        # Extract number
        num = None
        num_match = re.match(r'^(\d+)\.', text)
        if num_match:
            num = int(num_match.group(1))

        # Extract title within quotes
        # Pattern: **Author** (year). "Title." *Journal*
        title_pattern = r'\(\d{4}\)\s*\.\s*"([^"]+)"'
        title_match = re.search(title_pattern, text)
        if title_match:
            title = title_match.group(1).strip()
            # Clean title: remove trailing punctuation
            title = title.strip(' .,;:')
            return (num, title)

        # If no quotes found, try other patterns
        # Format: num. **Author** (year). Title. *Journal*
        # Extract content after (year) until *Journal*
        fallback_pattern = r'\(\d{4}\)\s*\.\s*([^.*]+?)\s*\.\s*\*'
        fallback_match = re.search(fallback_pattern, text)
        if fallback_match:
            title = fallback_match.group(1).strip()
            title = title.strip(' .,;:')
            return (num, title)

    # ===== Detect qwen format =====
    # Qwen format: contains "et al." and year pattern (year)
    qwen_pattern = r'.*et\s+al\.\s*\(\d{4}\)\s*\.\s*(.+?)\s*\.\s*\*'
    if is_qwen_format or re.search(r'et\s+al\.\s*\(\d{4}\)', text_lower):
        # Use original case text for extraction
        text = ' '.join(original_text.split())
        qwen_match = re.search(qwen_pattern, text, re.IGNORECASE)
        if qwen_match:
            title = qwen_match.group(1).strip()
            # Clean title: remove trailing punctuation
            title = title.strip(' .,;:')
            return (None, title)
        # If regex fails, try a more lenient pattern
        # Format: Author, et al. (year). Title. *Journal*
        # Extract content after (year) before the next period
        year_title_pattern = r'\(\d{4}\)\s*\.\s*([^.*]+?)\s*\.\s*\*'
        year_title_match = re.search(year_title_pattern, text, re.IGNORECASE)
        if year_title_match:
            title = year_title_match.group(1).strip()
            title = title.strip(' .,;:')
            return (None, title)

    # ===== Extract number =====
    num = None
    text = text_lower

    # Pattern: [cite: 1] or [1] or 1. or 1)
    num_pattern = r'^\[?(?:cite:\s*)?(\d+)[\]\.)\s]+'
    num_match = re.match(num_pattern, text)

    if num_match:
        num = int(num_match.group(1))
        # Remove number part
        text = text[num_match.end():].strip()

    # ===== Clean text =====

    # Remove URL (including URL: prefix)
    text = re.sub(r'\burl:\s*https?://\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove DOI
    text = re.sub(r'\bdoi:\s*\S+', '', text, flags=re.IGNORECASE)

    # Remove journal info: *journal name*, vol(issue), pages
    # e.g.: *Revista de Psicología*, 19(38), 41-62
    text = re.sub(r'\*[^*]+\*,?\s*\d+\(\d+\),\s*\d+-\d+', '', text)

    # Remove trailing [cite: num]
    text = re.sub(r'\[cite:\s*\d+\]\s*$', '', text)

    # Remove year: (1936–2024) or (2023) or year at end
    text = re.sub(r'\(\d{4}(?:[-–—]\d{4})?\)', '', text)
    text = re.sub(r',?\s*\d{4}(?:[-–—]\d{4})?\s*\.?\s*$', '', text)  # Remove trailing year

    # ===== Extract title (remove author names and journal names) =====

    # Strategy 1: Find common journal pattern "Journal Name. (Year). Title"
    # If matched, extract content before the first period as journal name
    journal_pattern = r'^([^.]+)\.\s*\(\d{4}\)\.\s*(.+)$'
    journal_match = re.match(journal_pattern, text)
    if journal_match:
        # In this case, first part is journal name, second is title
        # Keep the content before the first period
        text = journal_match.group(1).strip()
    else:
        # Strategy 2: If text has periods, analyze content before/after periods
        # Typically: Title. Author1, Author2, Author3
        # Or: Journal. Title
        parts = text.split('.')
        if len(parts) >= 2:
            # Check if second part looks like an author list (multiple comma-separated names)
            second_part = parts[1].strip()
            # If second part has multiple commas, likely an author list
            if second_part.count(',') >= 2:
                # First part is the title
                text = parts[0].strip()
            # If first part is short (<30 chars) and second part is longer, first part may be journal
            elif len(parts[0]) < 30 and len(second_part) > len(parts[0]):
                # Second part may be the title
                # But need to clean author names
                title_candidate = second_part
                # Remove possible author names (comma-separated person names)
                author_removal = re.sub(r'[,;]\s*[a-z]+\s+[a-z]+.*$', '', title_candidate)
                if author_removal:
                    text = author_removal.strip()
                else:
                    text = parts[0].strip()
            else:
                # Default: use first part
                text = parts[0].strip()

    # Strategy 3: Remove trailing author name patterns
    # e.g.: ", Author1, Author2" or ". Author1, Author2"
    text = re.sub(r'[.,]\s+[a-z]+\s+[a-z]+(?:,\s+[a-z]+\s+[a-z]+)+.*$', '', text)

    # Remove "and Author" pattern (only when commas present, to avoid removing "and" in titles)
    text = re.sub(r',\s+and\s+[a-z]+\s+[a-z]+.*$', '', text)

    # ===== Final cleanup =====

    # Remove extra punctuation and spaces
    text = text.strip(' .,;:')

    # Merge extra spaces
    text = ' '.join(text.split())

    # Remove consecutive periods
    text = re.sub(r'\.{2,}', '.', text)
    text = text.strip('.')
    
    return (num, text) 
    

def calculate_text_stats(text: str) -> Dict[str, int]:
    """
    Calculate text statistics.

    Args:
        text: Text content

    Returns:
        Dict: Statistics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'line_count': len(text.split('\n'))
    }


def robust_json_parse(response_text: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Robust JSON parsing function that handles parse failures with limited retries.

    Uses regex to extract content between the first { and last } (inclusive),
    handling cases where LLM adds extra text outside JSON. Supports up to
    `max_retries` attempts, with simple cleanup before each retry.

    Args:
        response_text: JSON response text
        max_retries: Maximum retry count (default 3)

    Returns:
        Optional[Dict]: Parsed dict, or None on failure
    """
    cleaned = response_text
    for attempt in range(1, max_retries + 1):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON fragment
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            if attempt < max_retries:
                # Basic cleanup before retry: remove wrapping code blocks, trim whitespace
                cleaned = cleaned.strip()
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'```$', '', cleaned)
                logging.warning(
                    f"JSON parse failed (attempt {attempt}/{max_retries-1}), retrying..."
                )
            else:
                logging.warning("Failed to parse JSON response after retries")
                logging.warning(f"Response: {cleaned[:500]}...")
                return None

# ==================== Core Cleaner Classes ====================

class ContentCleaner:
    """Content cleaner - responsible for cleaning outline and content."""

    def __init__(self, config: DataProcessingConfig):
        """
        Initialize the cleaner.

        Args:
            config: Data processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_outline(self, outline: List[List]) -> List[OutlineItem]:
        """
        Clean outline data.

        Args:
            outline: Raw outline [[level, title], ...]

        Returns:
            List[OutlineItem]: Cleaned outline
        """
        cleaned_outline = []

        for item in outline:
            try:
                outline_item = OutlineItem.from_list(item)

                # Clean title
                if self.config.normalize_outline:
                    outline_item.title = normalize_text(outline_item.title)
                    outline_item.title = remove_noise_patterns(outline_item.title)
                    outline_item.title = strip_leading_non_letters(outline_item.title)

                # Remove empty titles
                if self.config.remove_empty_titles:
                    if not outline_item.title or len(outline_item.title.strip()) < 2:
                        self.logger.debug(f"Skipping empty title at level {outline_item.level}")
                        continue

                cleaned_outline.append(outline_item)

            except Exception as e:
                self.logger.warning(f"Invalid outline item: {item}, error: {e}")
                continue

        self.logger.info(f"Cleaned outline: {len(outline)} -> {len(cleaned_outline)} items")
        return cleaned_outline

    def clean_content_sections(self, content_sections: List[Dict]) -> List[ContentSection]:
        """
        Clean content sections data.

        Args:
            content_sections: Raw content sections

        Returns:
            List[ContentSection]: Cleaned sections
        """
        cleaned_sections = []

        for section_data in content_sections:
            try:
                section = ContentSection.from_dict(section_data)

                # Clean content
                if self.config.normalize_content:
                    section.content = normalize_text(section.content)
                    section.content = remove_noise_patterns(section.content)
                    section.content = strip_leading_non_letters(section.content)
                    section.heading = normalize_text(section.heading)

                # Check content length
                if self.config.remove_short_sections:
                    word_count = len(section.content.split())
                    if word_count < self.config.min_section_words:
                        self.logger.debug(
                            f"Skipping short section '{section.heading}' ({word_count} words)"
                        )
                        continue

                # Calculate statistics
                if self.config.calculate_section_stats:
                    section.stats = calculate_text_stats(section.content)

                cleaned_sections.append(section)

            except Exception as e:
                self.logger.warning(f"Error cleaning section: {e}")
                continue

        self.logger.info(
            f"Cleaned sections: {len(content_sections)} -> {len(cleaned_sections)} sections"
        )
        return cleaned_sections


class ReferenceCleaner:
    """Reference cleaner - responsible for cleaning and normalizing references."""

    def __init__(self, config: DataProcessingConfig):
        """
        Initialize the reference cleaner.

        Args:
            config: Data processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize LLM client
        self.llm_client = None
        if self.config.llm_quality_check and LLM_AVAILABLE:
            # Read config from .env file (higher priority than config)
            api_key = os.environ.get("API_KEY") or self.config.llm_api_key
            base_url = os.environ.get("BASE_URL") or self.config.llm_api_base

            if api_key:
                self.llm_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                self.logger.info(f"LLM client initialized with model: {self.config.llm_model}")
                self.logger.info(f"Using base URL: {base_url}")
            else:
                self.logger.warning("LLM quality check enabled but no API key found (set API_KEY in .env)")
        elif self.config.llm_quality_check and not LLM_AVAILABLE:
            self.logger.warning("LLM quality check enabled but OpenAI library not available")

    def extract_reference_with_llm(self, reference_text: str) -> Tuple[int, str]:
        """
        Use LLM to extract reference number and title.

        Args:
            reference_text: Raw reference text

        Returns:
            Tuple[int, str]: (number, title), returns 0 if no number
        """
        if not self.llm_client:
            self.logger.warning("LLM client not available, falling back to regex extraction")
            return extract_reference_title(reference_text)
        
        try:
            prompt = f"""You are a reference parser. Extract the reference number and title from the following reference text.

Reference text:
{reference_text}

Please extract:
1. The reference number (if present, otherwise return 0)
2. The title of the paper/article

Return your answer in the following JSON format:
{{
    "number": <number as integer, 0 if not present>,
    "title": "<title as string>"
}}

Guidelines:
- Remove URLs, DOIs, author names, journal names, publication years, and other metadata
- Extract only the core title of the work
- If there's no explicit number, use 0
- Keep the title concise and clean
"""

            # Build request parameters
            request_params = {
                "model": self.config.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.config.llm_temperature,
                "response_format": {"type": "json_object"}
            }
            
            # If reasoning enabled, add extra_body
            if self.config.llm_enable_reasoning:
                request_params["extra_body"] = {"reasoning": {"enabled": False}}
            
            completion = self.llm_client.chat.completions.create(**request_params)
            
            # Parse LLM response
            response_text = completion.choices[0].message.content
            response_data = robust_json_parse(response_text)
            
            # If parsing failed, fall back to regex
            if response_data is None:
                self.logger.warning("JSON parsing failed, falling back to regex")
                return extract_reference_title(reference_text)
            
            number = response_data.get("number", 0)
            title = response_data.get("title", "").strip()
            
            # Ensure number is an integer
            if number is None or number == "":
                number = 0
            else:
                try:
                    number = int(number)
                except (ValueError, TypeError):
                    number = 0
            
            if not title:
                self.logger.warning(f"LLM returned empty title for reference: {reference_text[:100]}...")
                # Fall back to regex
                return extract_reference_title(reference_text)
            
            return (number, title)
            
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}, falling back to regex")
            return extract_reference_title(reference_text)
    
    def clean_references(self, reference_entries: List[ReferenceEntry], source_file: Optional[str] = None) -> List[ReferenceEntry]:
        """
        Clean reference list, extract titles and remove extra information.

        Args:
            reference_entries: Raw reference entry list
            source_file: Source file path (used to detect qwen or gemini format)

        Returns:
            List[ReferenceEntry]: Cleaned reference entry list
        """
        cleaned_refs = []
        seen_refs = set()  # For deduplication

        # Detect if references are gemini-generated (via file path or format)
        is_gemini = False
        if source_file and 'gemini' in source_file.lower():
            is_gemini = True
        elif reference_entries:
            # Check if first reference matches gemini format
            first_ref = reference_entries[0].text if reference_entries else ""
            if re.search(r'\d+\.\s*\*\*.*?\*\*\s*\(\d{4}\)\s*\.\s*"[^"]+"', first_ref):
                is_gemini = True

        # Detect if references are qwen-generated (via file path or format)
        is_qwen = False
        if not is_gemini:  # If already gemini, skip qwen check
            if source_file and 'qwen' in source_file.lower():
                is_qwen = True
            elif reference_entries and not is_gemini:
                # Check if first reference matches qwen format
                first_ref = reference_entries[0].text if reference_entries else ""
                if re.search(r'et\s+al\.\s*\(\d{4}\)', first_ref, re.IGNORECASE):
                    is_qwen = True

        for entry in reference_entries:
            # If normalization needed, re-extract title
            if self.config.normalize_references:
                # Choose extraction method based on config
                if self.config.llm_quality_check and self.llm_client:
                    # Use LLM to extract number and title
                    num, title = self.extract_reference_with_llm(entry.text)
                else:
                    # Use regex to extract number and title
                    # For qwen or gemini formats, use specialized extraction logic
                    num, title = extract_reference_title(entry.text, is_qwen_format=is_qwen, is_gemini_format=is_gemini)

                title = strip_leading_non_letters(title)

                # Build cleaned reference text
                if num is not None and num != 0 and self.config.keep_number:
                    cleaned_text = f"[{num}] {title}"
                else:
                    cleaned_text = title

                # Create new ReferenceEntry
                cleaned_entry = ReferenceEntry(
                    text=cleaned_text,
                    number=num if self.config.keep_number else None,
                    title=title
                )
            else:
                # No normalization, keep as-is (but strip whitespace)
                cleaned_entry = ReferenceEntry(
                    text=entry.text.strip(),
                    number=entry.number,
                    title=entry.title
                )

            # Remove empty references
            if not cleaned_entry.title or cleaned_entry.text == "[]":
                self.logger.debug("Skipping empty reference")
                continue

            # Deduplicate
            if self.config.remove_duplicate_refs:
                # Use lowercase title for deduplication
                ref_key = cleaned_entry.title.lower().strip()

                if not ref_key:
                    continue

                if ref_key in seen_refs:
                    self.logger.debug(f"Skipping duplicate reference: {ref_key[:50]}...")
                    continue
                seen_refs.add(ref_key)

            cleaned_refs.append(cleaned_entry)

        self.logger.info(f"Cleaned references: {len(reference_entries)} -> {len(cleaned_refs)} items")
        return cleaned_refs


class QualityChecker:
    """Quality checker - validates the quality of processed data."""

    def __init__(self, config: DataProcessingConfig):
        """
        Initialize the quality checker.

        Args:
            config: Data processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_data_quality(self, survey_data: SurveyData) -> Tuple[bool, List[str]]:
        """
        Check data quality.

        Args:
            survey_data: Survey data object

        Returns:
            Tuple[bool, List[str]]: (passed, list of issues)
        """
        issues = []

        # Check outline
        if len(survey_data.outline.items) < self.config.min_outline_items:
            issues.append(
                f"Outline too short: {len(survey_data.outline.items)} items "
                f"(min: {self.config.min_outline_items})"
            )
        
        # Check content
        if not survey_data.content.sections:
            issues.append("No content sections found")
        
        # Check references
        if len(survey_data.references.entries) < self.config.min_references:
            issues.append(
                f"Too few references: {len(survey_data.references.entries)} "
                f"(min: {self.config.min_references})"
            )
        
        # Check content length
        total_content = ' '.join([section.content for section in survey_data.content.sections])
        if len(total_content) < self.config.min_total_content_length:
            issues.append(
                f"Content too short: {len(total_content)} chars "
                f"(min: {self.config.min_total_content_length})"
            )
        
        # Check empty content sections
        empty_sections = sum(1 for section in survey_data.content.sections if not section.content.strip())
        if empty_sections > 0:
            issues.append(f"Found {empty_sections} empty content sections")
        
        passed = len(issues) == 0
        
        if not passed:
            self.logger.warning(f"Quality check failed: {len(issues)} issues found")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("Quality check passed")
        
        return passed, issues
    
    def generate_quality_report(self, survey_data: SurveyData) -> Dict[str, Any]:
        """
        Generate a detailed quality report.

        Args:
            survey_data: Survey data object

        Returns:
            Dict: Quality report
        """
        passed, issues = self.check_data_quality(survey_data)
        stats = survey_data.get_statistics()
        
        # Calculate additional metrics
        avg_section_length = (
            stats['total_chars'] / stats['section_count'] 
            if stats['section_count'] > 0 else 0
        )
        
        level_distribution = {}
        for item in survey_data.outline.items:
            level_distribution[item.level] = level_distribution.get(item.level, 0) + 1
        
        return {
            'passed': passed,
            'issues': issues,
            'statistics': stats,
            'metrics': {
                'avg_section_length': round(avg_section_length, 2),
                'outline_depth': max([item.level for item in survey_data.outline.items]) if survey_data.outline.items else 0,
                'level_distribution': level_distribution,
                'empty_sections': stats['section_count'] - stats['sections_with_content']
            }
        }


# ==================== Main Pipeline Class ====================

class DataProcessingPipeline:
    """
    Data processing pipeline main class.

    Responsible for coordinating the entire data processing workflow.
    """

    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Data processing configuration, uses defaults if None
        """
        self.config = config or DataProcessingConfig()
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.content_cleaner = ContentCleaner(self.config)
        self.reference_cleaner = ReferenceCleaner(self.config)
        self.quality_checker = QualityChecker(self.config)
        
        self.logger.info("Data Processing Pipeline initialized")
    
    def process_single_file(self, json_path: str, output_path: Optional[str] = None) -> SurveyData:
        """
        Process a single JSON file.

        Args:
            json_path: Input JSON file path
            output_path: Output JSON file path (optional)

        Returns:
            SurveyData: Processed data object
        """
        self.logger.info(f"Processing: {json_path}")
        
        try:
            # Step 1: Load raw JSON data
            self.logger.info("Step 1: Loading JSON data...")
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # Step 2: Convert to data object
            self.logger.info("Step 2: Parsing data structure...")
            survey_data = SurveyData.from_dict(raw_data)

            # Step 3: Clean outline
            self.logger.info("Step 3: Cleaning outline...")
            cleaned_outline_items = self.content_cleaner.clean_outline(
                survey_data.outline.to_list()
            )
            survey_data.outline = Outline(items=cleaned_outline_items)

            # Step 4: Clean content
            self.logger.info("Step 4: Cleaning content...")
            cleaned_content_sections = self.content_cleaner.clean_content_sections(
                survey_data.content.to_list()
            )
            survey_data.content = Content(sections=cleaned_content_sections)

            # Step 5: Clean references
            self.logger.info("Step 5: Cleaning references...")
            cleaned_ref_entries = self.reference_cleaner.clean_references(
                survey_data.references.entries,
                source_file=json_path
            )
            survey_data.references = References(entries=cleaned_ref_entries)

            # Step 6: Add/update metadata
            self.logger.info("Step 6: Adding metadata...")
            survey_data.metadata = {
                'source_file': json_path,
                'processed_date': datetime.now().isoformat(),
                'config': {
                    'normalize_outline': self.config.normalize_outline,
                    'normalize_content': self.config.normalize_content,
                    'normalize_references': self.config.normalize_references
                }
            }
            
            # Step 7: Quality check
            if self.config.enable_quality_check:
                self.logger.info("Step 7: Quality check...")
                quality_report = self.quality_checker.generate_quality_report(survey_data)
                survey_data.metadata['quality_check'] = quality_report
            
            # Step 8: Save results
            if output_path:
                self._save_json(survey_data, output_path)
            
            self.logger.info(f"Successfully processed: {json_path}")
            return survey_data
            
        except Exception as e:
            self.logger.error(f"Error processing {json_path}: {e}", exc_info=True)
            raise
    
    def convert_markdown_files(self, directory: str) -> Dict[str, Any]:
        """
        Check directory for Markdown files and convert if no corresponding JSON exists.

        Args:
            directory: Directory path to check

        Returns:
            Dict: Conversion result statistics
        """
        self.logger.info(f"Checking for markdown files in: {directory}")
        
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return {'total': 0, 'converted': 0, 'skipped': 0, 'failed': 0}
        
        # Find all .md files
        md_files = list(dir_path.rglob("*.md"))
        
        results = {
            'total': len(md_files),
            'converted': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
        
        for md_file in md_files:
            try:
                # Check if corresponding JSON file already exists
                json_file = md_file.with_suffix('').with_suffix('.json')
                if json_file.name.endswith('_split.json'):
                    json_file = md_file.with_name(md_file.stem + '_split.json')
                else:
                    json_file = md_file.with_name(md_file.stem + '_split.json')
                
                if json_file.exists() and not self.config.overwrite_original_json:
                    self.logger.debug(f"JSON already exists for: {md_file.name} (skipping, overwrite=False)")
                    results['skipped'] += 1
                    results['details'].append({
                        'file': str(md_file),
                        'status': 'skipped',
                        'reason': 'json_exists'
                    })
                    continue
                
                if json_file.exists() and self.config.overwrite_original_json:
                    self.logger.info(f"Overwriting existing JSON for: {md_file.name}")
                
                # Convert MD to JSON
                self.logger.info(f"Converting markdown to JSON: {md_file.name}")
                converter = MarkdownToJsonConverter(str(md_file))
                converter.parse()
                output_path = converter.save_json()
                
                results['converted'] += 1
                results['details'].append({
                    'file': str(md_file),
                    'status': 'converted',
                    'output': output_path
                })
                
            except Exception as e:
                self.logger.error(f"Failed to convert {md_file}: {e}")
                results['failed'] += 1
                results['details'].append({
                    'file': str(md_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        self.logger.info(
            f"Markdown conversion complete: {results['converted']} converted, "
            f"{results['skipped']} skipped, {results['failed']} failed"
        )
        
        return results
    
    def process_directory(self, 
                         input_dir: Optional[str] = None,
                         output_dir: Optional[str] = None,
                         pattern: str = "*_split.json",
                         auto_convert_markdown: Optional[bool] = None) -> Dict[str, Any]:
        """
        Batch process JSON files in a directory.

        Args:
            input_dir: Input directory (optional, uses config default)
            output_dir: Output directory (optional, uses config default)
            pattern: File matching pattern
            auto_convert_markdown: Whether to auto-convert Markdown files (None uses config default)

        Returns:
            Dict: Processing result statistics
        """
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or self.config.output_dir
        
        # Use config default if not specified
        if auto_convert_markdown is None:
            auto_convert_markdown = self.config.auto_convert_markdown
        
        self.logger.info(f"Batch processing directory: {input_dir}")
        
        # Step 0: Auto-convert Markdown files (if enabled)
        markdown_results = None
        if auto_convert_markdown:
            self.logger.info("Step 0: Checking and converting markdown files...")
            markdown_results = self.convert_markdown_files(input_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all JSON files
        input_path = Path(input_dir)
        json_files = list(input_path.rglob(pattern))
        
        self.logger.info(f"Found {len(json_files)} JSON files matching '{pattern}'")
        
        results = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        for json_file in json_files:
            try:
                # Build output path (preserving directory structure)
                relative_path = json_file.relative_to(input_path)
                output_path = Path(output_dir) / relative_path
                
                # Skip if output file exists and is newer
                if output_path.exists():
                    if output_path.stat().st_mtime > json_file.stat().st_mtime:
                        self.logger.info(f"Skipping (up-to-date): {json_file}")
                        results['skipped'] += 1
                        continue
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process file
                survey_data = self.process_single_file(str(json_file), str(output_path))
                
                # Get statistics
                stats = survey_data.get_statistics()
                quality_passed = survey_data.metadata.get('quality_check', {}).get('passed', True)
                
                results['success'] += 1
                results['details'].append({
                    'file': str(json_file),
                    'status': 'success',
                    'output': str(output_path),
                    'quality_passed': quality_passed,
                    'statistics': stats
                })
                
            except Exception as e:
                self.logger.error(f"Failed to process {json_file}: {e}")
                results['failed'] += 1
                results['details'].append({
                    'file': str(json_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save batch processing results
        summary_path = Path(output_dir) / 'processing_summary.json'
        summary_data = {
            'markdown_conversion': markdown_results,
            'json_processing': results
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"Batch processing complete: "
            f"{results['success']}/{results['total']} succeeded, "
            f"{results['failed']} failed, "
            f"{results['skipped']} skipped"
        )
        
        if markdown_results:
            self.logger.info(
                f"Markdown conversion: "
                f"{markdown_results['converted']} converted, "
                f"{markdown_results['skipped']} skipped, "
                f"{markdown_results['failed']} failed"
            )
        
        return summary_data
    
    def process_by_category(self, 
                           category: Optional[str] = None,
                           system: Optional[str] = None) -> Dict[str, Any]:
        """
        Process data by system and category.

        Args:
            category: Category name (optional, None means all, e.g. Biology, Computer Science)
            system: System name (optional, None means all, e.g. Autosurvey)

        Returns:
            Dict: Processing result statistics
        """
        systems = [system] if system else self.config.get_systems()
        
        all_results = {
            'systems_processed': 0,
            'categories_processed': 0,
            'total_files': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'by_system': {}
        }
        
        for sys in systems:
            self.logger.info(f"Processing system: {sys}")
            
            # Get categories under this system
            categories = [category] if category else self.config.get_categories_in_system(sys)
            
            system_results = {
                'categories_processed': 0,
                'total_files': 0,
                'success': 0,
                'failed': 0,
                'skipped': 0,
                'by_category': {}
            }
            
            for cat in categories:
                self.logger.info(f"  Processing category: {cat}")
                
                # Determine input/output directories (System/Category structure)
                cat_input_dir = os.path.join(self.config.input_dir, sys, cat)
                cat_output_dir = os.path.join(self.config.output_dir, sys, cat)
                
                if not os.path.exists(cat_input_dir):
                    self.logger.warning(f"Category directory not found: {cat_input_dir}")
                    continue
                
                # Process this category
                cat_results = self.process_directory(cat_input_dir, cat_output_dir)
                
                # Extract JSON processing results
                json_results = cat_results.get('json_processing', cat_results)
                
                system_results['categories_processed'] += 1
                system_results['total_files'] += json_results.get('total', 0)
                system_results['success'] += json_results.get('success', 0)
                system_results['failed'] += json_results.get('failed', 0)
                system_results['skipped'] += json_results.get('skipped', 0)
                system_results['by_category'][cat] = cat_results
            
            all_results['systems_processed'] += 1
            all_results['categories_processed'] += system_results['categories_processed']
            all_results['total_files'] += system_results['total_files']
            all_results['success'] += system_results['success']
            all_results['failed'] += system_results['failed']
            all_results['skipped'] += system_results['skipped']
            all_results['by_system'][sys] = system_results
        
        return all_results
    
    def _save_json(self, survey_data: SurveyData, output_path: str):
        """
        Save SurveyData as JSON file.

        Args:
            survey_data: Survey data object
            output_path: Output path
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(survey_data.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved to: {output_path}")


# ==================== Command Line Interface ====================

def main():
    """Command line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Survey Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python data_processing_pipeline.py input.json -o output.json
  
  # Process directory
  python data_processing_pipeline.py input_dir --batch -o output_dir
  
  # Process by category
  python data_processing_pipeline.py --category Biology -o output_dir
  
  # Process all with custom config
  python data_processing_pipeline.py --batch --config config.json
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input JSON file or directory')
    parser.add_argument('-o', '--output', help='Output JSON file or directory')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--category', help='Process specific category')
    parser.add_argument('--system', help='Process specific system')
    parser.add_argument('--config', help='Path to config JSON file')
    parser.add_argument('--overwrite-original-json', action='store_true',
                       help='Overwrite existing original JSON files when converting from Markdown')
    
    # Processing options
    parser.add_argument('--no-normalize-outline', action='store_true',
                       help='Disable outline normalization')
    parser.add_argument('--no-normalize-content', action='store_true',
                       help='Disable content normalization')
    parser.add_argument('--no-normalize-refs', action='store_true',
                       help='Disable reference normalization')
    parser.add_argument('--no-quality-check', action='store_true',
                       help='Disable quality check')
    
    # Threshold options
    parser.add_argument('--min-section-words', type=int, default=5,
                       help='Minimum section word count (default: 5)')
    parser.add_argument('--min-outline-items', type=int, default=2,
                       help='Minimum outline items (default: 2)')
    parser.add_argument('--min-references', type=int, default=1,
                       help='Minimum references (default: 1)')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            config = DataProcessingConfig(**config_dict)
    else:
        config = DataProcessingConfig()
    
    # Apply command line argument overrides
    if args.no_normalize_outline:
        config.normalize_outline = False
    if args.no_normalize_content:
        config.normalize_content = False
    if args.no_normalize_refs:
        config.normalize_references = False
    if args.no_quality_check:
        config.enable_quality_check = False
    if args.overwrite_original_json:
        config.overwrite_original_json = True
    
    config.min_section_words = args.min_section_words
    config.min_outline_items = args.min_outline_items
    config.min_references = args.min_references
    
    if args.output:
        config.output_dir = args.output
    if args.input and args.batch:
        config.input_dir = args.input
    
    # Create pipeline
    pipeline = DataProcessingPipeline(config)
    
    # Execute processing
    try:
        if args.category or args.system:
            # Process by category (if only system specified, processes all categories under it)
            results = pipeline.process_by_category(
                category=args.category,
                system=args.system
            )
            print(f"\n{'='*60}")
            print(f"Category Processing Summary")
            print(f"{'='*60}")
            print(f"Categories processed: {results['categories_processed']}")
            print(f"Total files: {results['total_files']}")
            print(f"Success: {results['success']}")
            print(f"Failed: {results['failed']}")
            print(f"Skipped: {results['skipped']}")
            
        elif args.batch:
            # Batch processing (supports filtering by system/category via config)
            if config.systems or config.categories:
                results = pipeline.process_by_category()
            else:
                if not args.input:
                    args.input = config.input_dir
                
                results = pipeline.process_directory(args.input, args.output)
            print(f"\n{'='*60}")
            print(f"Batch Processing Summary")
            print(f"{'='*60}")
            
            # Extract JSON processing results from nested structure
            json_results = results.get('json_processing', {})
            print(f"Total files: {json_results.get('total', 0)}")
            print(f"Success: {json_results.get('success', 0)}")
            print(f"Failed: {json_results.get('failed', 0)}")
            print(f"Skipped: {json_results.get('skipped', 0)}")
            
            # Display Markdown conversion results (if any)
            md_results = results.get('markdown_conversion')
            if md_results:
                print(f"\nMarkdown Conversion:")
                print(f"Converted: {md_results.get('converted', 0)}")
                print(f"Skipped: {md_results.get('skipped', 0)}")
                print(f"Failed: {md_results.get('failed', 0)}")
            
        else:
            # Single file processing
            if not args.input:
                parser.error("Input file is required for single file processing")
            
            survey_data = pipeline.process_single_file(args.input, args.output)
            
            print(f"\n{'='*60}")
            print(f"Processing Complete")
            print(f"{'='*60}")
            print(f"Source: {args.input}")
            if args.output:
                print(f"Output: {args.output}")
            
            stats = survey_data.get_statistics()
            print(f"\nStatistics:")
            print(f"  Outline items: {stats['outline_count']}")
            print(f"  Content sections: {stats['section_count']}")
            print(f"  References: {stats['reference_count']}")
            print(f"  Total words: {stats['total_words']}")
            
            if config.enable_quality_check and survey_data.metadata:
                qc = survey_data.metadata.get('quality_check', {})
                print(f"\nQuality Check: {'PASSED' if qc.get('passed') else 'FAILED'}")
                if not qc.get('passed'):
                    print("Issues:")
                    for issue in qc.get('issues', []):
                        print(f"  - {issue}")
        
        print(f"\n{'='*60}")
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
