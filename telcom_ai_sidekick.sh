# NEEDS FAKE WEB HOST AND APIs
# FUCKING FAKE THE FUCK OUT OF IT

#!/usr/bin/env python3
"""
Enhanced Telecom AI Coding Sidekick
Advanced offline-capable AI assistant for telecom development including:
- Kafka streaming and message processing
- SSH/Moshell network management
- O-RAN/srsRAN 5G/NR/LTE protocols
- Kubernetes orchestration
- GNU Radio SDR processing
- Complex physics calculations
- Ubuntu 22.04 system optimization
"""

import os
import sys
import time
import json
import hashlib
import sqlite3
import subprocess
import threading
import traceback
import ast
import queue
import re
import math
import cmath
import signal
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import logging
import pickle
import gzip
import configparser

# External dependencies with fallbacks
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available - file watching disabled")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.prompt import Prompt, Confirm
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich not available - using basic console output")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available - complex physics calculations limited")

try:
    import scipy
    from scipy import signal as scipy_signal
    from scipy.fft import fft, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available - advanced signal processing disabled")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available - YAML parsing disabled")

# Configure logging with enhanced format for telecom debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('telecom_ai_sidekick.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if RICH_AVAILABLE:
    console = Console()
else:
    class BasicConsole:
        def print(self, *args, **kwargs):
            print(*args)
        def clear(self):
            os.system('clear' if os.name == 'posix' else 'cls')
    console = BasicConsole()

# ----------------------------
# TELECOM DOMAIN MODELS
# ----------------------------
class TelecomTechnology(Enum):
    """Supported telecom technologies"""
    LTE = "LTE"
    NR_5G = "5G-NR"
    ORAN = "O-RAN"
    SRSRAN = "srsRAN"
    KAFKA = "Kafka"
    GNU_RADIO = "GNU Radio"
    KUBERNETES = "Kubernetes"
    SSH_MOSHELL = "SSH/Moshell"
    COMPLEX_PHYSICS = "Complex Physics"

@dataclass
class TelecomConfig:
    """Telecom-specific configuration"""
    # 5G/LTE Parameters
    frequency_bands: List[int] = field(default_factory=lambda: [1, 3, 7, 20, 28, 78])  # Common 5G bands
    bandwidth_mhz: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
    numerology: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # 5G NR numerology
    
    # O-RAN Configuration
    oran_components: List[str] = field(default_factory=lambda: [
        "O-RU", "O-DU", "O-CU-CP", "O-CU-UP", "RIC", "SMO"
    ])
    oran_interfaces: List[str] = field(default_factory=lambda: [
        "7-2x", "F1-C", "F1-U", "E1", "X2", "Xn", "A1", "E2", "O1"
    ])
    
    # srsRAN Configuration
    srsran_components: List[str] = field(default_factory=lambda: [
        "srsUE", "srsENB", "srsEPC", "srsGNB", "srs5GC"
    ])
    
    # Kafka Configuration
    kafka_topics: List[str] = field(default_factory=lambda: [
        "cell-metrics", "ue-events", "network-kpis", "alarms", "traces"
    ])
    kafka_brokers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    
    # GNU Radio
    sample_rates: List[int] = field(default_factory=lambda: [
        1000000, 2000000, 5000000, 10000000, 20000000  # Common SDR sample rates
    ])
    
    # Kubernetes
    k8s_namespaces: List[str] = field(default_factory=lambda: [
        "ran", "core", "edge", "monitoring", "kafka"
    ])

@dataclass
class NetworkAnalysis:
    """Network protocol analysis results"""
    technology: TelecomTechnology
    protocol_violations: List[str]
    performance_metrics: Dict[str, float]
    optimization_suggestions: List[str]
    complexity_score: float
    timestamp: datetime

@dataclass
class PhysicsCalculation:
    """Complex physics calculation results"""
    calculation_type: str
    input_parameters: Dict[str, Any]
    results: Dict[str, Any]
    units: Dict[str, str]
    validity_range: Dict[str, Tuple[float, float]]
    timestamp: datetime

# ----------------------------
# ENHANCED CONFIGURATION
# ----------------------------
@dataclass
class EnhancedConfig:
    # Original configuration
    model_name: str = "offline-fallback"  # No external models for offline
    embedding_model: str = "local-tfidf"
    max_model_length: int = 2048
    temperature: float = 0.7
    
    # Enhanced watch configuration for telecom
    watch_paths: List[str] = # ----------------------------
# LANGUAGE LEARNING SYSTEM
# ----------------------------

class LanguageLearningSystem:
    """System for learning new programming languages and frameworks"""
    def __init__(self, config: Config, db_manager: DatabaseManager, language_registry: LanguageRegistry):
        self.config = config
        self.db_manager = db_manager
        self.language_registry = language_registry
        self.learning_buffer = defaultdict(list)  # Buffer for learning examples by language
        self.pattern_extractors = {}
        self.confidence_scores = defaultdict(float)
        
    def observe_code(self, content: str, file_path: str, user_feedback: Optional[Dict] = None):
        """Observe code and learn patterns"""
        if not self.config.enable_language_learning:
            return
        
        # Detect or learn language
        language = self.language_registry.detect_language(file_path, content)
        
        if not language:
            # Try to learn a new language
            potential_language = self._attempt_language_learning(content, file_path)
            if potential_language:
                language = potential_language
        
        if language:
            # Store learning example
            example = {
                'content': content,
                'file_path': file_path,
                'timestamp': datetime.now(),
                'user_feedback': user_feedback or {}
            }
            
            self.learning_buffer[language.name].append(example)
            
            # Update patterns if we have enough examples
            if len(self.learning_buffer[language.name]) >= self.config.min_examples_for_learning:
                self._update_language_patterns(language.name)
    
    def _attempt_language_learning(self, content: str, file_path: str) -> Optional[LanguageDefinition]:
        """Attempt to learn a new language from code examples"""
        extension = Path(file_path).suffix.lower()
        
        # Extract potential patterns
        patterns = self._extract_patterns_from_content(content)
        keywords = self._extract_keywords_from_content(content)
        
        # Check if this looks like a coherent language
        if len(keywords) >= 5 and len(patterns) >= 3:
            # Create tentative language definition
            tentative_language = LanguageDefinition(
                name=f"Unknown_{extension[1:].upper()}" if extension else "Unknown_Language",
                extensions=[extension] if extension else [],
                keywords=keywords,
                patterns=patterns,
                complexity_rules={pattern: 1 for pattern in patterns.keys()},
                best_practices=[],
                common_issues=[],
                optimization_patterns=[]
            )
            
            logger.info(f"Learning new language: {tentative_language.name}")
            return tentative_language
        
        return None
    
    def _extract_patterns_from_content(self, content: str) -> Dict[str, str]:
        """Extract common patterns from code content"""
        patterns = {}
        
        # Function definitions (various patterns)
        if re.search(r'\bfunc\s+\w+\s*\(', content):
            patterns['function'] = r'func\s+(\w+)\s*\('
        elif re.search(r'\bdef\s+\w+\s*\(', content):
            patterns['function'] = r'def\s+(\w+)\s*\('
        elif re.search(r'\w+\s*\([^)]*\)\s*{', content):
            patterns['function'] = r'(\w+)\s*\([^)]*\)\s*{'
        
        # Class/struct definitions
        if re.search(r'\bclass\s+\w+', content):
            patterns['class'] = r'class\s+(\w+)'
        elif re.search(r'\bstruct\s+\w+', content):
            patterns['struct'] = r'struct\s+(\w+)'
        
        # Import/include statements
        if re.search(r'\bimport\s+', content):
            patterns['import'] = r'import\s+([\w.]+)'
        elif re.search(r'#include\s*<', content):
            patterns['include'] = r'#include\s*<([^>]+)>'
        elif re.search(r'\buse\s+', content):
            patterns['use'] = r'use\s+([\w:]+)'
        
        # Control structures
        if re.search(r'\bif\s*\(', content):
            patterns['conditional'] = r'if\s*\([^)]+\)'
        if re.search(r'\bfor\s*\(', content):
            patterns['loop'] = r'for\s*\([^)]*\)'
        elif re.search(r'\bwhile\s*\(', content):
            patterns['loop'] = r'while\s*\([^)]+\)'
        
        return patterns
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """Extract potential keywords from code content"""
        # Common programming keywords that might appear
        potential_keywords = [
            'function', 'func', 'def', 'class', 'struct', 'interface', 'enum',
            'if', 'else', 'elif', 'for', 'while', 'do', 'switch', 'case',
            'return', 'yield', 'break', 'continue', 'try', 'catch', 'finally',
            'throw', 'throws', 'import', 'from', 'include', 'use', 'require',
            'public', 'private', 'protected', 'static', 'const', 'let', 'var',
            'async', 'await', 'promise', 'thread', 'mutex', 'atomic',
            'template', 'generic', 'trait', 'protocol', 'impl', 'extend'
        ]
        
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in potential_keywords:
            if re.search(r'\b' + keyword + r'\b', content_lower):
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _update_language_patterns(self, language_name: str):
        """Update patterns for a language based on accumulated examples"""
        examples = self.learning_buffer[language_name]
        
        if len(examples) < self.config.min_examples_for_learning:
            return
        
        # Analyze patterns across examples
        pattern_frequency = defaultdict(int)
        keyword_frequency = defaultdict(int)
        
        for example in examples:
            content = example['content']
            
            # Count pattern occurrences
            for pattern_name, pattern in self._extract_patterns_from_content(content).items():
                pattern_frequency[pattern_name] += len(re.findall(pattern, content))
            
            # Count keyword occurrences
            for keyword in self._extract_keywords_from_content(content):
                keyword_frequency[keyword] += 1
        
        # Update language definition if it exists
        if language_name in self.language_registry.languages:
            language = self.language_registry.languages[language_name]
            
            # Add new patterns that appear frequently
            for pattern_name, frequency in pattern_frequency.items():
                if frequency >= len(examples) * 0.5:  # Appears in 50% of examples
                    if pattern_name not in language.patterns:
                        # This is a simplified pattern - would need more sophisticated extraction
                        language.patterns[pattern_name] = f"\\b{pattern_name}\\b"
            
            # Add new keywords that appear frequently
            for keyword, frequency in keyword_frequency.items():
                if frequency >= len(examples) * 0.3 and keyword not in language.keywords:
                    language.keywords.append(keyword)
            
            logger.info(f"Updated patterns for language: {language_name}")
    
    def add_user_defined_language(self, name: str, definition_file: str):
        """Add a user-defined language from a configuration file"""
        try:
            if definition_file.endswith('.json'):
                with open(definition_file, 'r') as f:
                    lang_data = json.load(f)
            elif definition_file.endswith('.yaml') and YAML_AVAILABLE:
                with open(definition_file, 'r') as f:
                    lang_data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported definition file format")
            
            # Create language definition
            language = LanguageDefinition(
                name=lang_data.get('name', name),
                extensions=lang_data.get('extensions', []),
                keywords=lang_data.get('keywords', []),
                patterns=lang_data.get('patterns', {}),
                complexity_rules=lang_data.get('complexity_rules', {}),
                best_practices=lang_data.get('best_practices', []),
                common_issues=lang_data.get('common_issues', []),
                optimization_patterns=lang_data.get('optimization_patterns', [])
            )
            
            self.language_registry.register_language(language)
            logger.info(f"Added user-defined language: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add user-defined language {name}: {e}")
    
    def export_learned_language(self, language_name: str, output_file: str):
        """Export a learned language definition to a file"""
        if language_name not in self.language_registry.languages:
            raise ValueError(f"Language {language_name} not found")
        
        language = self.language_registry.languages[language_name]
        lang_data = asdict(language)
        
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(lang_data, f, indent=2)
            elif output_file.endswith('.yaml') and YAML_AVAILABLE:
                with open(output_file, 'w') as f:
                    yaml.dump(lang_data, f, default_flow_style=False)
            else:
                raise ValueError("Unsupported output file format")
            
            logger.info(f"Exported language definition for {language_name} to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export language definition: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about language learning"""
        stats = {
            'supported_languages': len(self.language_registry.languages),
            'learning_buffer_sizes': {lang: len(examples) 
                                    for lang, examples in self.learning_buffer.items()},
            'confidence_scores': dict(self.confidence_scores),
            'total_examples': sum(len(examples) for examples in self.learning_buffer.values())
        }
        
        return stats

# ----------------------------
# ENHANCED TELECOM SIDEKICK WITH LANGUAGE LEARNING
# ----------------------------

class TelecomSidekick:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.db_manager = DatabaseManager(self.config.db_path)
        
        # Initialize language learning system
        self.language_registry = LanguageRegistry()
        self.language_learning = LanguageLearningSystem(
            self.config, self.db_manager, self.language_registry
        )
        self.adaptive_analyzer = AdaptiveLanguageAnalyzer(self.language_registry)
        
        # Existing components
        self.analysis_engine = OfflineAnalysisEngine(self.config, self.db_manager)
        self.physics_engine = PhysicsEngine(self.config)
        self.system_monitor = SystemMonitor(self.config)
        self.self_learning_manager = SelfLearningManager(self.config)
        self.observer = Observer() if WATCHDOG_AVAILABLE else None
        self.handler = None
        self.running = False
        self.stats = {
            'files_analyzed': 0,
            'technologies_detected': defaultdict(int),
            'languages_learned': 0,
            'errors_detected': 0,
            'suggestions_made': 0,
            'start_time': None
        }
        
    def analyze_code_with_language_learning(self, content: str, file_path: str, 
                                          user_feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze code with language learning capabilities"""
        # Let the language learning system observe this code
        self.language_learning.observe_code(content, file_path, user_feedback)
        
        # Perform adaptive analysis
        language_analysis = self.adaptive_analyzer.analyze_code(content, file_path)
        
        # Perform existing telecom-specific analysis
        telecom_analysis = self.analysis_engine.analyze_code(content, file_path)
        
        # Combine results
        combined_analysis = {
            **telecom_analysis,
            'language_analysis': language_analysis,
            'detected_language': language_analysis.get('language', 'Unknown'),
            'language_confidence': self.language_learning.confidence_scores.get(
                language_analysis.get('language', 'Unknown'), 0.0
            )
        }
        
        return combined_analysis
    
    def add_custom_language(self, name: str, definition: Union[str, Dict]):
        """Add a custom language definition"""
        if isinstance(definition, str):
            # Assume it's a file path
            self.language_learning.add_user_defined_language(name, definition)
        else:
            # Assume it's a dictionary definition
            language = LanguageDefinition(
                name=name,
                extensions=definition.get('extensions', []),
                keywords=definition.get('keywords', []),
                patterns=definition.get('patterns', {}),
                complexity_rules=definition.get('complexity_rules', {}),
                best_practices=definition.get('best_practices', []),
                common_issues=definition.get('common_issues', []),
                optimization_patterns=definition.get('optimization_patterns', [])
            )
            self.language_registry.register_language(language)
        
        self.stats['languages_learned'] += 1
        logger.info(f"Added custom language: {name}")
    
    def export_language_knowledge(self, language_name: str, output_file: str):
        """Export learned knowledge about a language"""
        self.language_learning.export_learned_language(language_name, output_file)
    
    def get_language_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about language learning"""
        return self.language_learning.get_learning_stats()
    
    def _interactive_language_commands(self):
        """Interactive commands for language learning"""
        commands = {
            'languages': self._show_supported_languages,
            'learn': self._interactive_language_learning,
            'add_lang': self._add_custom_language_interactive,
            'export_lang': self._export_language_interactive,
            'lang_stats': self._show_language_stats
        }
        return commands
    
    def _show_supported_languages(self):
        """Show currently supported languages"""
        if not RICH_AVAILABLE:
            print("\nSupported Languages:")
            for name, lang in self.language_registry.languages.items():
                print(f"  {name}: {', '.join(lang.extensions)}")
            return
        
        lang_table = Table(title="Supported Programming Languages", border_style="green")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Extensions", style="yellow")
        lang_table.add_column("Keywords", style="green")
        lang_table.add_column("Patterns", style="magenta")
        
        for name, lang in self.language_registry.languages.items():
            extensions = ", ".join(lang.extensions)
            keywords = f"{len(lang.keywords)} keywords"
            patterns = f"{len(lang.patterns)} patterns"
            lang_table.add_row(name, extensions, keywords, patterns)
        
        console.print(lang_table)
    
    def _interactive_language_learning(self):
        """Interactive language learning session"""
        if not RICH_AVAILABLE:
            print("\nInteractive Language Learning")
            file_path = input("Enter path to example file: ").strip()
            if not Path(file_path).exists():
                print("File not found")
                return
        else:
            console.print("[cyan]Interactive Language Learning Session[/cyan]")
            file_path = Prompt.ask("Enter path to example file")
            if not Path(file_path).exists():
                console.print("[red]File not found[/red]")
                return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze with current knowledge
            analysis = self.adaptive_analyzer.analyze_code(content, file_path)
            
            if RICH_AVAILABLE:
                console.print(f"[green]Detected language: {analysis.get('language', 'Unknown')}[/green]")
                
                if analysis.get('language') == 'Unknown':
                    if Confirm.ask("Would you like to help train a new language?"):
                        self._train_new_language_interactive(content, file_path)
            else:
                print(f"Detected language: {analysis.get('language', 'Unknown')}")
                if analysis.get('language') == 'Unknown':
                    response = input("Would you like to help train a new language? (y/n): ")
                    if response.lower() == 'y':
                        self._train_new_language_interactive(content, file_path)
        
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
    
    def _train_new_language_interactive(self, content: str, file_path: str):
        """Interactive training for a new language"""
        if RICH_AVAILABLE:
            console.print("[yellow]Training new language...[/yellow]")
            lang_name = Prompt.ask("What language is this?")
            
            # Show detected patterns
            patterns = self.language_learning._extract_patterns_from_content(content)
            keywords = self.language_learning._extract_keywords_from_content(content)
            
            console.print(f"[cyan]Detected patterns: {list(patterns.keys())}[/cyan]")
            console.print(f"[cyan]Detected keywords: {keywords[:10]}[/cyan]")
            
            if Confirm.ask("Do these look correct?"):
                # Create language definition
                extension = Path(file_path).suffix.lower()
                language_def = LanguageDefinition(
                    name=lang_name,
                    extensions=[extension] if extension else [],
                    keywords=keywords,
                    patterns=patterns,
                    complexity_rules={pattern: 1 for pattern in patterns.keys()},
                    best_practices=[],
                    common_issues=[],
                    optimization_patterns=[]
                )
                
                self.language_registry.register_language(language_def)
                console.print(f"[green]Successfully learned language: {lang_name}[/green]")
        else:
            print("Training new language...")
            lang_name = input("What language is this? ").strip()
            
            patterns = self.language_learning._extract_patterns_from_content(content)
            keywords = self.language_learning._extract_keywords_from_content(content)
            
            print(f"Detected patterns: {list(patterns.keys())}")
            print(f"Detected keywords: {keywords[:10]}")
            
            confirm = input("Do these look correct? (y/n): ")
            if confirm.lower() == 'y':
                extension = Path(file_path).suffix.lower()
                language_def = LanguageDefinition(
                    name=lang_name,
                    extensions=[extension] if extension else [],
                    keywords=keywords,
                    patterns=patterns,
                    complexity_rules={pattern: 1 for pattern in patterns.keys()},
                    best_practices=[],
                    common_issues=[],
                    optimization_patterns=[]
                )
                
                self.language_registry.register_language(language_def)
                print(f"Successfully learned language: {lang_name}")
    
    def _add_custom_language_interactive(self):
        """Interactive custom language addition"""
        if RICH_AVAILABLE:
            console.print("[cyan]Add Custom Language[/cyan]")
            choice = Prompt.ask("Load from file or create interactively?", 
                              choices=["file", "interactive"], default="file")
            
            if choice == "file":
                file_path = Prompt.ask("Enter language definition file path")
                lang_name = Prompt.ask("Enter language name")
                try:
                    self.add_custom_language(lang_name, file_path)
                    console.print(f"[green]Successfully added language: {lang_name}[/green]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            else:
                # Interactive creation
                lang_name = Prompt.ask("Language name")
                extensions = Prompt.ask("File extensions (comma-separated)", default=".ext").split(",")
                keywords = Prompt.ask("Keywords (comma-separated)", default="").split(",")
                
                definition = {
                    'extensions': [ext.strip() for ext in extensions],
                    'keywords': [kw.strip() for kw in keywords if kw.strip()],
                    'patterns': {},
                    'complexity_rules': {},
                    'best_practices': [],
                    'common_issues': [],
                    'optimization_patterns': []
                }
                
                self.add_custom_language(lang_name, definition)
                console.print(f"[green]Successfully created language: {lang_name}[/green]")
        else:
            print("Add Custom Language")
            choice = input("Load from file or create interactively? (file/interactive): ").strip()
            
            if choice == "file":
                file_path = input("Enter language definition file path: ").strip()
                lang_name = input("Enter language name: ").strip()
                try:
                    self.add_custom_language(lang_name, file_path)
                    print(f"Successfully added language: {lang_name}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                lang_name = input("Language name: ").strip()
                extensions = input("File extensions (comma-separated): ").split(",")
                keywords = input("Keywords (comma-separated): ").split(",")
                
                definition = {
                    'extensions': [ext.strip() for ext in extensions],
                    'keywords': [kw.strip() for kw in keywords if kw.strip()],
                    'patterns': {},
                    'complexity_rules': {},
                    'best_practices': [],
                    'common_issues': [],
                    'optimization_patterns': []
                }
                
                self.add_custom_language(lang_name, definition)
                print(f"Successfully created language: {lang_name}")
    
    def _export_language_interactive(self):
        """Interactive language export"""
        if not RICH_AVAILABLE:
            print("Export Language Knowledge")
            languages = list(self.language_registry.languages.keys())
            for i, lang in enumerate(languages, 1):
                print(f"  {i}. {lang}")
            
            try:
                choice = int(input("Choose language number: ")) - 1
                if 0 <= choice < len(languages):
                    lang_name = languages[choice]
                    output_file = input("Output file path: ").strip()
                    self.export_language_knowledge(lang_name, output_file)
                    print(f"Exported {lang_name} to {output_file}")
            except (ValueError, IndexError):
                print("Invalid choice")
            return
        
        languages = list(self.language_registry.languages.keys())
        lang_name = Prompt.ask("Choose language to export", choices=languages)
        output_file = Prompt.ask("Output file path", default=f"{lang_name.lower()}_definition.json")
        
        try:
            self.export_language_knowledge(lang_name, output_file)
            console.print(f"[green]Exported {lang_name} to {output_file}[/green]")
        except Exception as e:
            console.print(f"[red]Export failed: {e}[/red]")
    
    def _show_language_stats(self):
        """Show language learning statistics"""
        stats = self.get_language_learning_stats()
        
        if not RICH_AVAILABLE:
            print("\nLanguage Learning Statistics:")
            print(f"Supported Languages: {stats['supported_languages']}")
            print(f"Total Examples: {stats['total_examples']}")
            print("Learning Buffer Sizes:")
            for lang, size in stats['learning_buffer_sizes'].items():
                print(f"  {lang}: {size}")
            return
        
        stats_table = Table(title="Language Learning Statistics", border_style="blue")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Supported Languages", str(stats['supported_languages']))
        stats_table.add_row("Total Examples", str(stats['total_examples']))
        stats_table.add_row("Languages Learning", str(len(stats['learning_buffer_sizes'])))
        
        console.print(stats_table)
        
        if stats['learning_buffer_sizes']:
            buffer_table = Table(title="Learning Buffer Status", border_style="green")
            buffer_table.add_column("Language", style="cyan")
            buffer_table.add_column("Examples", style="yellow")
            buffer_table.add_column("Confidence", style="green")
            
            for lang, size in stats['learning_buffer_sizes'].items():
                confidence = stats['confidence_scores'].get(lang, 0.0)
                buffer_table.add_row(lang, str(size), f"{confidence:.2f}")
            
            console.print(buffer_table)

# ----------------------------
# ENHANCED INTERACTIVE COMMANDS
# ----------------------------

    def _run_interactive_loop(self):
        """Enhanced interactive command loop with language learning"""
        base_commands = {
            'help': self._show_help,
            'stats': self._show_stats,
            'config': self._show_config,
            'analyze': self._manual_analyze,
            'recent': self._show_recent_analyses,
            'physics': self._physics_calculator,
            'kafka': self._kafka_simulation,
            'system': self._show_system_status,
            'telecom': self._show_telecom_info,
            'clear': lambda: console.clear() if RICH_AVAILABLE else os.system('clear'),
            'quit': lambda: None,
            'exit': lambda: None
        }
        
        # Add language learning commands
        language_commands = self._interactive_language_commands()
        commands = {**base_commands, **language_commands}
        
        while self.running:
            try:
                if RICH_AVAILABLE:
                    command = Prompt.ask(
                        "\n[bold cyan]Enhanced Telecom AI Sidekick[/bold cyan]",
                        default="help"
                    ).lower().strip()
                else:
                    command = input("\nEnhanced Telecom AI Sidekick> ").lower().strip()
                    if not command:
                        command = "help"
                
                if command in ['quit', 'exit']:
                    break
                elif command in commands:
                    commands[command]()
                else:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        console.print("Type 'help' for available commands")
                    else:
                        print(f"Unknown command: {command}")
                        print("Type 'help' for available commands")
                        
            except (EOFError, KeyboardInterrupt):
                break

    def _show_help(self):
        """Display enhanced help information with language learning commands"""
        if not RICH_AVAILABLE:
            print("\nAvailable Commands:")
            print("Core Commands:")
            print("  help        - Show this help message")
            print("  stats       - Display analysis statistics")
            print("  config      - Show current configuration")
            print("  analyze     - Manually analyze a specific file")
            print("  recent      - Show recent analysis results")
            print("  physics     - Physics calculator for telecom")
            print("  kafka       - Kafka simulation and analysis")
            print("  system      - Show system status")
            print("  telecom     - Show telecom-specific information")
            print("\nLanguage Learning Commands:")
            print("  languages   - Show supported languages")
            print("  learn       - Interactive language learning")
            print("  add_lang    - Add custom language definition")
            print("  export_lang - Export language knowledge")
            print("  lang_stats  - Show language learning statistics")
            print("\nUtility Commands:")
            print("  clear       - Clear the terminal screen")
            print("  quit/exit   - Stop the AI sidekick")
            return
        
        help_table = Table(title="Enhanced Commands with Language Learning", border_style="blue")
        help_table.add_column("Command", style="cyan", width=15)
        help_table.add_column("Description", style="white")
        
        # Core commands
        help_table.add_row("help", "Show this help message")
        help_table.add_row("stats", "Display enhanced analysis statistics")
        help_table.add_row("config", "Show current configuration")
        help_table.add_row("analyze <file>", "Manually analyze a specific file")
        help_table.add_row("recent", "Show recent analysis results")
        help_table.add_row("physics", "Interactive physics calculator")
        help_table.add_row("kafka", "Kafka simulation and analysis tools")
        help_table.add_row("system", "Show system resource status")
        help_table.add_row("telecom", "Show telecom-specific information")
        
        # Language learning commands
        help_table.add_row("", "")  # Separator
        help_table.add_row("languages", "Show all supported languages")
        help_table.add_row("learn", "Interactive language learning session")
        help_table.add_row("add_lang", "Add custom language definition")
        help_table.add_row("export_lang", "Export learned language knowledge")
        help_table.add_row("lang_stats", "Show language learning statistics")
        
        # Utility commands
        help_table.add_row("", "")  # Separator
        help_table.add_row("clear", "Clear the terminal screen")
        help_table.add_row("quit/exit", "Stop the enhanced AI sidekick")
        
        console.print(help_table)

# ----------------------------
# EXAMPLE LANGUAGE DEFINITIONS
# ----------------------------

def create_example_language_definitions():
    """Create example language definition files"""
    
    # Example Zig language definition
    zig_definition = {
        "name": "Zig",
        "extensions": [".zig"],
        "keywords": ["fn", "struct", "enum", "union", "const", "var", "pub", "export", "extern", "packed", "test"],
        "patterns": {
            "function": r"fn\s+(\w+)\s*\(",
            "struct": r"(const|var)\s+(\w+)\s*=\s*struct",
            "test": r"test\s+\"([^\"]+)\"",
            "comptime": r"comptime\s+",
            "error": r"error\s*\{",
            "defer": r"defer\s+"
        },
        "complexity_rules": {
            "comptime": 1,
            "error": 2,
            "defer": 1,
            "union": 2
        },
        "best_practices": [
            "Use defer for cleanup operations",
            "Prefer compile-time evaluation with comptime",
            "Handle errors explicitly with error unions",
            "Use packed structs for memory-mapped I/O"
        ],
        "common_issues": [
            "Memory leaks from missing defer statements",
            "Runtime errors from unchecked error unions",
            "Performance issues from unnecessary runtime operations"
        ],
        "optimization_patterns": [
            "Use comptime for compile-time calculations",
            "Prefer ArrayList over slices when size varies",
            "Use packed structs for bit manipulation"
        ]
    }
    
    # Example VHDL definition for hardware description
    vhdl_definition = {
        "name": "VHDL",
        "extensions": [".vhd", ".vhdl"],
        "keywords": ["entity", "architecture", "component", "signal", "variable", "process", "begin", "end", "port", "generic"],
        "patterns": {
            "entity": r"entity\s+(\w+)\s+is",
            "architecture": r"architecture\s+(\w+)\s+of\s+(\w+)",
            "process": r"(\w+)?\s*:\s*process\s*\(",
            "signal": r"signal\s+(\w+)\s*:",
            "component": r"component\s+(\w+)",
            "clock": r"rising_edge\s*\(\s*(\w+)\s*\)",
            "reset": r"(\w*reset\w*)",
        },
        "complexity_rules": {
            "process": 2,
            "nested_if": 1,
            "state_machine": 3,
            "clock_domain": 2
        },
        "best_practices": [
            "Use synchronous design practices",
            "Implement proper reset strategies",
            "Use standard logic types consistently",
            "Document clock domains clearly"
        ],
        "common_issues": [
            "Clock domain crossing violations",
            "Latch inference from incomplete assignments",
            "Setup and hold timing violations"
        ],
        "optimization_patterns": [
            "Pipeline critical paths for timing",
            "Use block RAM for large memory structures",
            "Minimize logic levels in combinational paths"
        ]
    }
    
    # Save example definitions
    os.makedirs("language_definitions", exist_ok=True)
    
    with open("language_definitions/zig.json", "w") as f:
        json.dump(zig_definition, f, indent=2)
    
    with open("language_definitions/vhdl.json", "w") as f:
        json.dump(vhdl_definition, f, indent=2)
    
    print("Created example language definitions in language_definitions/")

# ----------------------------
# MAIN EXECUTION
# ----------------------------

def main():
    """Main entry point with enhanced language learning capabilities"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Offline Telecom AI Coding Sidekick with Language Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --setup                    # Interactive configuration setup
  %(prog)s --check                    # Check system requirements
  %(prog)s --install                  # Install dependencies
  %(prog)s --analyze file.py          # Analyze a specific file
  %(prog)s --watch /path/to/code      # Watch directory for changes
  %(prog)s --physics path_loss        # Perform physics calculation
  %(prog)s --learn-lang file.ext      # Learn new language from example
  %(prog)s --add-lang lang.json       # Add language from definition file
  %(prog)s --create-examples          # Create example language definitions
  
Language Learning Features:
  - Automatically learns new programming languages from code examples
  - Supports custom language definitions via JSON/YAML files
  - Adapts analysis patterns based on observed code structures
  - Exports learned knowledge for sharing and backup
        """
    )
    
    # Existing arguments
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--setup", "-s", action="store_true", help="Run interactive configuration setup")
    parser.add_argument("--check", action="store_true", help="Check system requirements")
    parser.add_argument("--install", action="store_true", help="Install required dependencies")
    parser.add_argument("--analyze", "-a", help="Analyze a specific file")
    parser.add_argument("--watch", "-w", nargs="+", help="Directories to watch for changes")
    parser.add_argument("--physics", "-p", help="Perform physics calculation")
    parser.add_argument("--params", help="Parameters for physics calculation (JSON string)")
    
    # Language learning arguments
    parser.add_argument("--learn-lang", help="Learn new language from example file")
    parser.add_argument("--add-lang", help="Add language from definition file")
    parser.add_argument("--export-lang", help="Export language definition")
    parser.add_argument("--lang-name", help="Language name for operations")
    parser.add_argument("--output", "-o", help="Output file for exports")
    parser.add_argument("--create-examples", action="store_true", help="Create example language definitions")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--version", action="version", version="Enhanced Telecom AI Sidekick v3.1.0")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.create_examples:
            create_example_language_definitions()
            return
        
        if args.check:
            check_system_requirements()
            return
        
        if args.install:
            install_dependencies()
            return
        
        if args.setup:
            config_path = setup_configuration()
            if RICH_AVAILABLE:
                console.print(f"[green]Setup complete! Use --config {config_path} to load your configuration[/green]")
            else:
                print(f"Setup complete! Use --config {config_path} to load your configuration")
            return
        
        # Create sidekick instance
        sidekick = TelecomSidekick(args.config)
        
        # Handle language learning operations
        if args.learn_lang:
            if not Path(args.learn_lang).exists():
                if RICH_AVAILABLE:
                    console.print(f"[red]File not found: {args.learn_lang}[/red]")
                else:
                    print(f"File not found: {args.learn_lang}")
                return
            
            try:
                with open(args.learn_lang, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if RICH_AVAILABLE:
                    console.print("[cyan]Learning language from example...[/cyan]")
                else:
                    print("Learning language from example...")
                
                # Try to learn the language
                sidekick.language_learning.observe_code(content, args.learn_lang)
                
                if RICH_AVAILABLE:
                    console.print("[green]Language learning session complete[/green]")
                else:
                    print("Language learning session complete")
                
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Language learning failed: {e}[/red]")
                else:
                    print(f"Language learning failed: {e}")
            return
        
        if args.add_lang:
            if not args.lang_name:
                if RICH_AVAILABLE:
                    console.print("[red]Language name required (use --lang-name)[/red]")
                else:
                    print("Language name required (use --lang-name)")
                return
            
            try:
                sidekick.add_custom_language(args.lang_name, args.add_lang)
                if RICH_AVAILABLE:
                    console.print(f"[green]Successfully added language: {args.lang_name}[/green]")
                else:
                    print(f"Successfully added language: {args.lang_name}")
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Failed to add language: {e}[/red]")
                else:
                    print(f"Failed to add language: {e}")
            return
        
        if args.export_lang:
            if not args.lang_name:
                if RICH_AVAILABLE:
                    console.print("[red]Language name required (use --lang-name)[/red]")
                else:
                    print("Language name required (use --lang-name)")
                return
            
            output_file = args.output or f"{args.lang_name.lower()}_definition.json"
            
            try:
                sidekick.export_language_knowledge(args.lang_name, output_file)
                if RICH_AVAILABLE:
                    console.print(f"[green]Exported {args.lang_name} to {output_file}[/green]")
                else:
                    print(f"Exported {args.lang_name} to {output_file}")
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Export failed: {e}[/red]")
                else:
                    print(f"Export failed: {e}")
            return
        
        # Handle existing operations with enhanced language analysis
        if args.analyze:
            if not Path(args.analyze).exists():
                if RICH_AVAILABLE:
                    console.print(f"[red]File not found: {args.analyze}[/red]")
                else:
                    print(f"File not found: {args.analyze}")
                return
            
            try:
                with open(args.analyze, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if RICH_AVAILABLE:
                    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                        task = progress.add_task("Analyzing code with language learning...", total=None)
                        analysis = sidekick.analyze_code_with_language_learning(content, args.analyze)
                        progress.update(task, description="✅ Analysis complete")
                else:
                    print("Analyzing code with language learning...")
                    analysis = sidekick.analyze_code_with_language_learning(content, args.analyze)
                
                # Display enhanced results
                if RICH_AVAILABLE:
                    results_table = Table(title=f"Enhanced Analysis: {Path(args.analyze).name}", border_style="green")
                    results_table.add_column("Metric", style="cyan")
                    results_table.add_column("Value", style="yellow")
                    
                    results_table.add_row("Detected Language", analysis.get('detected_language', 'Unknown'))
                    results_table.add_row("Language Confidence", f"{analysis.get('language_confidence', 0):.2f}")
                    results_table.add_row("Technologies", ", ".join([t.value for t in analysis.get('technologies', [])]))
                    results_table.add_row("Complexity Score", f"{analysis.get('complexity', 0):.3f}")
                    results_table.add_row("Analysis Time", f"{analysis.get('analysis_time', 0):.3f}s")
                    
                    console.print(results_table)
                    
                    # Show language-specific analysis
                    lang_analysis = analysis.get('language_analysis', {})
                    if lang_analysis.get('patterns_found'):
                        console.print("\n[bold cyan]Language Patterns Found:[/bold cyan]")
                        for pattern, count in lang_analysis['patterns_found'].items():
                            console.print(f"  {pattern}: {count}")
                    
                    if analysis.get('optimizations'):
                        console.print("\n[bold green]💡 Optimization Suggestions:[/bold green]")
                        for i, suggestion in enumerate(analysis['optimizations'][:5], 1):
                            console.print(f"  {i}. {suggestion}")
                else:
                    print(f"Enhanced Analysis Results for {Path(args.analyze).name}:")
                    print(f"  Language: {analysis.get('detected_language', 'Unknown')}")
                    print(f"  Confidence: {analysis.get('language_confidence', 0):.2f}")
                    print(f"  Technologies: {[t.value for t in analysis.get('technologies', [])]}")
                    print(f"  Complexity: {analysis.get('complexity', 0):.3f}")
                    
                    if analysis.get('optimizations'):
                        print("  Optimization Suggestions:")
                        for suggestion in analysis['optimizations'][:3]:
                            print(f"    - {suggestion}")
                
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Analysis failed: {e}[/red]")
                else:
                    print(f"Analysis failed: {e}")
        
        elif args.physics:
            # Existing physics calculation code...
            try:
                params = json.loads(args.params) if args.params else {}
                
                if RICH_AVAILABLE:
                    console.print(f"[cyan]Performing {args.physics} calculation...[/cyan]")
                else:
                    print(f"Performing {args.physics} calculation...")
                
                calculation = sidekick.calculate_physics(args.physics, params)
                
                if RICH_AVAILABLE:
                    calc_table = Table(title=f"Physics Calculation: {args.physics}", border_style="blue")
                    calc_table.add_column("Parameter", style="cyan")
                    calc_table.add_column("Value", style="yellow")
                    calc_table.add_column("Unit", style="green")
                    
                    for key, value in calculation.results.items():
                        unit = calculation.units.get(key, "")
                        if isinstance(value, float):
                            value_str = f"{value:.3f}"
                        else:
                            value_str = str(value)
                        calc_table.add_row(key, value_str, unit)
                    
                    console.print(calc_table)
                else:
                    print(f"Physics Calculation Results ({args.physics}):")
                    for key, value in calculation.results.items():
                        unit = calculation.units.get(key, "")
                        print(f"  {key}: {value} {unit}")
                
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Physics calculation failed: {e}[/red]")
                else:
                    print(f"Physics calculation failed: {e}")
        
        else:
            # Start enhanced interactive mode with language learning
            if RICH_AVAILABLE:
                console.print("\n[bold green]Enhanced Telecom AI Sidekick with Language Learning[/bold green]")
                console.print("[cyan]New features: Dynamic language learning, adaptive analysis, custom language support[/cyan]")
            else:
                print("\nEnhanced Telecom AI Sidekick with Language Learning")
                print("New features: Dynamic language learning, adaptive analysis, custom language support")
            
            sidekick.start()
            
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        else:
            print("\nInterrupted by user")
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Fatal error: {e}[/red]")
        else:
            print(f"Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

# ----------------------------
# USAGE EXAMPLES AND DOCUMENTATION
# ----------------------------

"""
USAGE EXAMPLES FOR LANGUAGE LEARNING:

1. Learn a new language from example code:
   python telecom_ai_sidekick.py --learn-lang example.rs --lang-name Rust

2. Add a custom language definition:
   python telecom_ai_sidekick.py --add-lang rust_definition.json --lang-name Rust

3. Export learned language knowledge:
   python telecom_ai_sidekick.py --export-lang --lang-name Rust --output rust_learned.json

4. Interactive language learning session:
   python telecom_ai_sidekick.py
   > learn

5. Analyze code with enhanced language detection:
   python telecom_ai_sidekick.py --analyze complex_algorithm.zig

LANGUAGE DEFINITION FILE FORMAT (JSON):
{
  "name": "LanguageName",
  "extensions": [".ext1", ".ext2"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "patterns": {
    "function": "regex_pattern_for_functions",
    "class": "regex_pattern_for_classes",
    "import": "regex_pattern_for_imports"
  },
  "complexity_rules": {
    "pattern_name": complexity_weight_integer
  },
  "best_practices": [
    "Best practice description 1",
    "Best practice description 2"
  ],
  "common_issues": [
    "Common issue description 1",
    "Common issue description 2"
  ],
  "optimization_patterns": [
    "Optimization suggestion 1",
    "Optimization suggestion 2"
  ]
}

SUPPORTED LEARNING MODES:

1. AUTOMATIC LEARNING:
   - Detects new languages from file patterns and keywords
   - Extracts common constructs automatically
   - Builds pattern database through observation
   - Improves suggestions over time

2. SUPERVISED LEARNING:
   - User provides language examples with feedback
   - System learns from corrections and improvements
   - Builds confidence scores for different patterns
   - Adapts to specific coding styles and preferences

3. DEFINITION-BASED LEARNING:
   - Load complete language definitions from files
   - Support for JSON and YAML definition formats
   - Immediate full-featured language support
   - Easy sharing of language definitions

LANGUAGE CATEGORIES SUPPORTED:

- General Purpose: Python, JavaScript, Java, C++, C, Go, Rust, Zig
- System Programming: C, C++, Rust, Zig, Assembly
- Web Development: JavaScript, TypeScript, HTML, CSS, PHP
- Functional: Haskell, Lisp, Erlang, Elixir
- Domain-Specific: SQL, VHDL, Verilog, MATLAB, R
- Configuration: YAML, JSON, TOML, XML, INI
- Build Systems: Makefile, CMake, Dockerfile, Kubernetes YAML
- Hardware Description: VHDL, Verilog, SystemVerilog
- Smart Contracts: Solidity, Vyper
- Emerging: WebAssembly, Move, Cairo

ADAPTIVE FEATURES:

1. PATTERN EVOLUTION:
   - Patterns improve based on code exposure
   - False positives decrease over time
   - New constructs automatically detected
   - Context-aware pattern matching

2. SUGGESTION REFINEMENT:
   - Suggestions become more relevant with usage
   - User feedback improves recommendation quality
   - Technology-specific advice integration
   - Performance-oriented optimization hints

3. COMPLEXITY ASSESSMENT:
   - Language-specific complexity metrics
   - Adaptive thresholds based on codebase
   - Context-aware complexity scoring
   - Maintenance-focused complexity analysis

INTEGRATION WITH EXISTING FEATURES:

- Telecom-specific patterns for networking code
- Physics calculations for signal processing languages
- System optimization for performance-critical languages
- Security analysis for smart contract languages
- Hardware optimization for HDL languages

ADVANCED USAGE:

1. Custom Analyzer Classes:
   Create specialized analyzers for specific languages or domains
   
2. Pattern Mining:
   Automatically discover new patterns in large codebases
   
3. Cross-Language Analysis:
   Detect patterns that span multiple languages in mixed projects
   
4. Style Adaptation:
   Learn and adapt to specific team or project coding styles
   
5. Domain Specialization:
   Focus learning on specific domains (telecom, ML, embedded, etc.)
"""
    extensions: List[str] = None
    ignore_patterns: List[str] = None
    max_file_size_mb: int = 50  # Larger for telecom logs
    
    # Telecom-specific extensions
    telecom_extensions: List[str] = field(default_factory=lambda: [
        ".cfg", ".conf", ".yaml", ".yml", ".xml", ".json",  # Config files
        ".log", ".trace", ".pcap", ".cap",                   # Network captures
        ".grc", ".py", ".cpp", ".c", ".h",                  # GNU Radio & C/C++
        ".k8s", ".dockerfile", ".helm",                      # Kubernetes
        ".sql", ".proto", ".thrift",                        # Data formats
        ".m", ".octave", ".mat",                            # MATLAB/Octave
        ".r", ".R",                                         # R statistical computing
        ".asn1", ".der", ".pem"                             # Telecom protocols
    ])
    
    # Analysis Configuration
    max_lines_analysis: int = 1000  # Increased for telecom files
    similarity_threshold: float = 0.75
    cache_duration_hours: int = 24
    offline_mode: bool = True
    
    # Execution Configuration
    enable_auto_execution: bool = False
    execution_timeout: int = 60  # Longer for telecom processing
    max_execution_attempts: int = 3
    sandbox_mode: bool = True
    
    # Database Configuration
    db_path: str = "telecom_ai_sidekick.db"
    
    # Telecom Configuration
    telecom_config: TelecomConfig = field(default_factory=TelecomConfig)
    
    # System Configuration
    ubuntu_version: str = "22.04"
    enable_system_optimization: bool = True
    monitor_system_resources: bool = True
    
    def __post_init__(self):
        if self.watch_paths is None:
            self.watch_paths = [
                os.path.expanduser("~/"),
                os.path.expanduser("~/telecom"),
                os.path.expanduser("~/projects"),
                os.path.expanduser("~/srsran"),
                os.path.expanduser("~/gnuradio"),
                "/opt/srsran",
                "/usr/local/share/gnuradio",
                "/etc/kubernetes",
                os.getcwd()
            ]
        
        if self.extensions is None:
            self.extensions = [
                # Programming languages
                ".py", ".cpp", ".c", ".h", ".hpp", ".cs", ".java",
                ".js", ".ts", ".go", ".rs", ".rb", ".php", ".swift",
                ".scala", ".r", ".m", ".pl", ".sh", ".bash",
                # Web and config
                ".html", ".css", ".xml", ".json", ".yaml", ".yml",
                ".toml", ".ini", ".cfg", ".conf", ".env",
                # Documentation
                ".md", ".rst", ".txt", ".doc", ".pdf",
                # Jupyter and data
                ".ipynb", ".csv", ".tsv", ".parquet",
                # Build systems
                ".mk", ".cmake", ".dockerfile", ".docker-compose.yml",
            ] + self.telecom_extensions
        
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                "__pycache__", ".git", ".svn", ".hg", ".bzr",
                "node_modules", ".venv", "venv", "env", ".env",
                ".pytest_cache", ".mypy_cache", ".coverage",
                "*.pyc", "*.pyo", "*.pyd", ".DS_Store",
                "*.log", "*.tmp", "*.temp", "*.bak", "*.swp",
                ".idea", ".vscode", "*.o", "*.so", "*.dll",
                "build", "dist", "target", "cmake-build-*",
                # Telecom specific ignores
                "*.pcap", "*.cap", "*.trace",  # Can be very large
                "logs/*", "traces/*", "captures/*"
            ]

# ----------------------------
# OFFLINE ANALYSIS ENGINE
# ----------------------------
class OfflineAnalysisEngine:
    """Offline analysis engine for telecom code without external AI models"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.patterns = self._load_patterns()
        self.telecom_analyzers = {
            TelecomTechnology.LTE: self._analyze_lte_code,
            TelecomTechnology.NR_5G: self._analyze_5g_nr_code,
            TelecomTechnology.ORAN: self._analyze_oran_code,
            TelecomTechnology.SRSRAN: self._analyze_srsran_code,
            TelecomTechnology.KAFKA: self._analyze_kafka_code,
            TelecomTechnology.GNU_RADIO: self._analyze_gnuradio_code,
            TelecomTechnology.KUBERNETES: self._analyze_kubernetes_code,
            TelecomTechnology.SSH_MOSHELL: self._analyze_ssh_moshell_code,
            TelecomTechnology.COMPLEX_PHYSICS: self._analyze_physics_code
        }
    
    def _load_patterns(self) -> Dict[str, List[Dict]]:
        """Load analysis patterns for different technologies"""
        return {
            "lte_keywords": [
                {"pattern": r"\bENB\b", "description": "eNodeB reference"},
                {"pattern": r"\bUE\b", "description": "User Equipment reference"},
                {"pattern": r"\bEPC\b", "description": "Evolved Packet Core"},
                {"pattern": r"\bRRC\b", "description": "Radio Resource Control"},
                {"pattern": r"\bPDCP\b", "description": "Packet Data Convergence Protocol"},
                {"pattern": r"\bRLC\b", "description": "Radio Link Control"},
                {"pattern": r"\bMAC\b", "description": "Medium Access Control"},
                {"pattern": r"\bPHY\b", "description": "Physical Layer"},
                {"pattern": r"\bS1\b", "description": "S1 Interface"},
                {"pattern": r"\bX2\b", "description": "X2 Interface"},
            ],
            "5g_keywords": [
                {"pattern": r"\bgNB\b", "description": "5G NodeB"},
                {"pattern": r"\b5GC\b", "description": "5G Core"},
                {"pattern": r"\bNR\b", "description": "New Radio"},
                {"pattern": r"\bAMF\b", "description": "Access and Mobility Management Function"},
                {"pattern": r"\bSMF\b", "description": "Session Management Function"},
                {"pattern": r"\bUPF\b", "description": "User Plane Function"},
                {"pattern": r"\bNSSF\b", "description": "Network Slice Selection Function"},
                {"pattern": r"\bPCF\b", "description": "Policy Control Function"},
                {"pattern": r"\bNRF\b", "description": "Network Repository Function"},
                {"pattern": r"\bUDM\b", "description": "Unified Data Management"},
            ],
            "oran_keywords": [
                {"pattern": r"\bO-RU\b", "description": "O-RAN Radio Unit"},
                {"pattern": r"\bO-DU\b", "description": "O-RAN Distributed Unit"},
                {"pattern": r"\bO-CU\b", "description": "O-RAN Central Unit"},
                {"pattern": r"\bRIC\b", "description": "RAN Intelligent Controller"},
                {"pattern": r"\bSMO\b", "description": "Service Management and Orchestration"},
                {"pattern": r"\bxApp\b", "description": "RIC xApp"},
                {"pattern": r"\brApp\b", "description": "RIC rApp"},
                {"pattern": r"\b7-2x\b", "description": "O-RAN Fronthaul Interface"},
                {"pattern": r"\bE2\b", "description": "E2 Interface"},
                {"pattern": r"\bA1\b", "description": "A1 Interface"},
            ],
            "kafka_keywords": [
                {"pattern": r"\bKafkaProducer\b", "description": "Kafka Producer"},
                {"pattern": r"\bKafkaConsumer\b", "description": "Kafka Consumer"},
                {"pattern": r"\btopic\b", "description": "Kafka Topic"},
                {"pattern": r"\bpartition\b", "description": "Kafka Partition"},
                {"pattern": r"\boffset\b", "description": "Kafka Offset"},
                {"pattern": r"\bbroker\b", "description": "Kafka Broker"},
                {"pattern": r"\bzookeeper\b", "description": "Zookeeper"},
                {"pattern": r"\bserializer\b", "description": "Message Serializer"},
                {"pattern": r"\bdeserializer\b", "description": "Message Deserializer"},
            ],
            "gnuradio_keywords": [
                {"pattern": r"\bgr\.", "description": "GNU Radio module"},
                {"pattern": r"\btop_block\b", "description": "GNU Radio Top Block"},
                {"pattern": r"\busrp\b", "description": "USRP Device"},
                {"pattern": r"\bsamp_rate\b", "description": "Sample Rate"},
                {"pattern": r"\bfilter\b", "description": "Signal Filter"},
                {"pattern": r"\bfft\b", "description": "Fast Fourier Transform"},
                {"pattern": r"\bmodulation\b", "description": "Signal Modulation"},
                {"pattern": r"\bdemodulation\b", "description": "Signal Demodulation"},
            ],
            "kubernetes_keywords": [
                {"pattern": r"\bapiVersion\b", "description": "Kubernetes API Version"},
                {"pattern": r"\bkind:\s*Pod\b", "description": "Kubernetes Pod"},
                {"pattern": r"\bkind:\s*Service\b", "description": "Kubernetes Service"},
                {"pattern": r"\bkind:\s*Deployment\b", "description": "Kubernetes Deployment"},
                {"pattern": r"\bnamespace\b", "description": "Kubernetes Namespace"},
                {"pattern": r"\bkubectl\b", "description": "Kubernetes CLI"},
                {"pattern": r"\bhelm\b", "description": "Helm Package Manager"},
            ],
            "ssh_moshell_keywords": [
                {"pattern": r"\bssh\s+", "description": "SSH Command"},
                {"pattern": r"\bmoshell\b", "description": "Ericsson Moshell"},
                {"pattern": r"\blt\s+all\b", "description": "Moshell list command"},
                {"pattern": r"\bget\s+\w+\b", "description": "Moshell get command"},
                {"pattern": r"\bset\s+\w+\b", "description": "Moshell set command"},
                {"pattern": r"\bManagedElement\b", "description": "Network Element"},
                {"pattern": r"\bEUtranCellFDD\b", "description": "LTE Cell"},
                {"pattern": r"\bNRCellDU\b", "description": "5G NR Cell"},
            ],
            "physics_keywords": [
                {"pattern": r"\bcomplex\b", "description": "Complex numbers"},
                {"pattern": r"\bnumpy\b", "description": "NumPy scientific computing"},
                {"pattern": r"\bscipy\b", "description": "SciPy scientific library"},
                {"pattern": r"\bfft\b", "description": "Fast Fourier Transform"},
                {"pattern": r"\bmatrix\b", "description": "Matrix operations"},
                {"pattern": r"\beigenvalue\b", "description": "Eigenvalue computation"},
                {"pattern": r"\bfourier\b", "description": "Fourier analysis"},
                {"pattern": r"\bwave\b", "description": "Wave equations"},
                {"pattern": r"\bfrequency\b", "description": "Frequency domain"},
            ]
        }
    
    def analyze_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze code content for telecom-specific patterns"""
        results = {
            "technologies_detected": [],
            "suggestions": [],
            "warnings": [],
            "errors": [],
            "complexity_metrics": {},
            "network_analysis": None,
            "physics_analysis": None
        }
        
        # Detect technologies
        for tech in TelecomTechnology:
            if self._detect_technology(content, tech):
                results["technologies_detected"].append(tech)
                
                # Run technology-specific analysis
                analyzer = self.telecom_analyzers.get(tech)
                if analyzer:
                    tech_results = analyzer(content, file_path)
                    results["suggestions"].extend(tech_results.get("suggestions", []))
                    results["warnings"].extend(tech_results.get("warnings", []))
                    results["errors"].extend(tech_results.get("errors", []))
        
        # General code analysis
        results["complexity_metrics"] = self._analyze_complexity(content, file_path)
        results["suggestions"].extend(self._generate_general_suggestions(content, file_path))
        
        return results
    
    def _detect_technology(self, content: str, tech: TelecomTechnology) -> bool:
        """Detect if content contains specific technology patterns"""
        tech_patterns = {
            TelecomTechnology.LTE: "lte_keywords",
            TelecomTechnology.NR_5G: "5g_keywords", 
            TelecomTechnology.ORAN: "oran_keywords",
            TelecomTechnology.KAFKA: "kafka_keywords",
            TelecomTechnology.GNU_RADIO: "gnuradio_keywords",
            TelecomTechnology.KUBERNETES: "kubernetes_keywords",
            TelecomTechnology.SSH_MOSHELL: "ssh_moshell_keywords",
            TelecomTechnology.COMPLEX_PHYSICS: "physics_keywords"
        }
        
        pattern_key = tech_patterns.get(tech)
        if not pattern_key or pattern_key not in self.patterns:
            return False
        
        patterns = self.patterns[pattern_key]
        matches = 0
        
        for pattern_info in patterns:
            if re.search(pattern_info["pattern"], content, re.IGNORECASE):
                matches += 1
        
        # Require at least 2 matches to consider technology present
        return matches >= 2
    
    def _analyze_lte_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze LTE-specific code"""
        suggestions = []
        warnings = []
        errors = []
        
        # Check for common LTE patterns and best practices
        if "UE" in content and "attach" in content.lower():
            suggestions.append("Consider implementing UE attach retry mechanisms for robustness")
        
        if "RRC" in content and "measurement" in content.lower():
            suggestions.append("Ensure RRC measurement configurations follow 3GPP specifications")
        
        if re.search(r"frequency.*[0-9]+", content, re.IGNORECASE):
            suggestions.append("Validate frequency bands against operator license ranges")
        
        # Check for potential issues
        if "hardcoded" in content.lower() or re.search(r"\b\d{9,}\b", content):
            warnings.append("Potential hardcoded values detected - consider configuration files")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": errors
        }
    
    def _analyze_5g_nr_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze 5G NR-specific code"""
        suggestions = []
        warnings = []
        errors = []
        
        # 5G NR specific checks
        if "numerology" in content.lower():
            suggestions.append("Ensure numerology selection is optimized for use case (eMBB/URLLC/mMTC)")
        
        if "beam" in content.lower():
            suggestions.append("Consider beamforming optimization for mmWave frequencies")
        
        if "slice" in content.lower():
            suggestions.append("Implement proper network slice isolation and QoS enforcement")
        
        # Check for 5G core integration
        if any(func in content for func in ["AMF", "SMF", "UPF"]):
            suggestions.append("Ensure proper service-based architecture (SBA) implementation")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": errors
        }
    
    def _analyze_oran_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze O-RAN-specific code"""
        suggestions = []
        warnings = []
        
        if "xApp" in content or "rApp" in content:
            suggestions.append("Ensure RIC app follows O-RAN Alliance specifications")
            suggestions.append("Implement proper E2 interface message handling")
        
        if "7-2x" in content or "fronthaul" in content.lower():
            suggestions.append("Optimize fronthaul transport for latency requirements")
        
        if "vendor" in content.lower() and "agnostic" in content.lower():
            suggestions.append("Verify vendor-agnostic implementation follows O-RAN standards")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": []
        }
    
    def _analyze_srsran_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze srsRAN-specific code"""
        suggestions = []
        warnings = []
        
        if "srsUE" in content:
            suggestions.append("Configure UE parameters for realistic testing scenarios")
        
        if "srsENB" in content or "srsGNB" in content:
            suggestions.append("Optimize scheduler parameters for target performance")
        
        if ".conf" in file_path or "config" in content.lower():
            suggestions.append("Validate configuration against hardware capabilities")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": []
        }
    
    def _analyze_kafka_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Kafka-specific code"""
        suggestions = []
        warnings = []
        errors = []
        
        # Producer best practices
        if "KafkaProducer" in content:
            suggestions.append("Configure producer with appropriate acks, retries, and batching")
            if "acks=0" in content:
                warnings.append("acks=0 may result in message loss")
        
        # Consumer best practices
        if "KafkaConsumer" in content:
            suggestions.append("Implement proper offset management and error handling")
            if "auto_offset_reset" not in content:
                warnings.append("Consider explicit auto_offset_reset configuration")
        
        # Topic configuration
        if "create_topic" in content or "NewTopic" in content:
            suggestions.append("Consider partitioning strategy for scalability")
        
        # Error handling
        if "except" not in content and ("Producer" in content or "Consumer" in content):
            warnings.append("Add proper exception handling for Kafka operations")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": errors
        }
    
    def _analyze_gnuradio_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze GNU Radio-specific code"""
        suggestions = []
        warnings = []
        
        if "samp_rate" in content:
            suggestions.append("Ensure sample rate matches hardware capabilities")
        
        if "usrp" in content.lower():
            suggestions.append("Configure USRP gain and frequency settings appropriately")
        
        if "filter" in content and "taps" in content:
            suggestions.append("Optimize filter tap count for performance vs. accuracy")
        
        if ".grc" in file_path:
            suggestions.append("Consider converting to Python for better version control")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": []
        }
    
    def _analyze_kubernetes_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Kubernetes-specific code"""
        suggestions = []
        warnings = []
        errors = []
        
        # Resource management
        if "resources:" not in content and "kind: Pod" in content:
            warnings.append("Consider adding resource limits and requests")
        
        # Security
        if "privileged: true" in content:
            warnings.append("Avoid privileged containers when possible")
        
        if "runAsRoot" not in content and "securityContext" in content:
            suggestions.append("Explicitly configure runAsRoot for security")
        
        # High availability
        if "replicas: 1" in content:
            suggestions.append("Consider multiple replicas for high availability")
        
        # Probes
        if "livenessProbe" not in content and "kind: Deployment" in content:
            suggestions.append("Add liveness and readiness probes")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": errors
        }
    
    def _analyze_ssh_moshell_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze SSH/Moshell-specific code"""
        suggestions = []
        warnings = []
        
        if "ssh" in content and "password" in content.lower():
            warnings.append("Avoid hardcoded passwords, use SSH keys instead")
        
        if "moshell" in content:
            suggestions.append("Implement error handling for moshell command failures")
            suggestions.append("Use batch operations for efficiency")
        
        if "ManagedElement" in content:
            suggestions.append("Validate MO (Managed Object) paths before operations")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": []
        }
    
    def _analyze_physics_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze complex physics calculations"""
        suggestions = []
        warnings = []
        errors = []
        
        # Check for numerical stability
        if "1/" in content or "division" in content.lower():
            suggestions.append("Check for division by zero and numerical stability")
        
        # Complex number handling
        if "complex" in content or "1j" in content:
            suggestions.append("Ensure proper complex number precision for calculations")
        
        # Matrix operations
        if "matrix" in content or "numpy.dot" in content:
            suggestions.append("Consider using optimized BLAS libraries for large matrices")
        
        # FFT operations
        if "fft" in content.lower():
            suggestions.append("Ensure FFT input size is power of 2 for optimal performance")
        
        # Units and validation
        if any(unit in content.lower() for unit in ["hz", "mhz", "ghz", "db", "dbm"]):
            suggestions.append("Validate unit consistency throughout calculations")
        
        return {
            "suggestions": suggestions,
            "warnings": warnings,
            "errors": errors
        }
    
    def _analyze_complexity(self, content: str, file_path: str) -> Dict[str, float]:
        """Analyze code complexity"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Basic complexity metrics
        metrics = {
            'lines_of_code': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'complexity': 1,  # McCabe complexity
            'nesting_depth': 0
        }
        
        # Calculate cyclomatic complexity
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
        for line in lines:
            for keyword in complexity_keywords:
                metrics['complexity'] += line.count(keyword)
        
        # Calculate nesting depth
        current_depth = 0
        max_depth = 0
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(':') and any(kw in stripped for kw in ['if', 'for', 'while', 'def', 'class', 'try', 'with']):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                pass  # Same level
            elif line and not line.startswith(' ') and not line.startswith('\t'):
                current_depth = 0
        
        metrics['nesting_depth'] = max_depth
        return metrics
    
    def _generate_general_suggestions(self, content: str, file_path: str) -> List[str]:
        """Generate general code improvement suggestions"""
        suggestions = []
        lines = content.split('\n')
        
        # Check for TODO/FIXME comments
        for i, line in enumerate(lines, 1):
            if any(marker in line.upper() for marker in ['TODO', 'FIXME', 'HACK', 'XXX']):
                suggestions.append(f"Address TODO/FIXME comment at line {i}")
        
        # Check for long lines
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
        if long_lines:
            suggestions.append(f"Consider breaking long lines (found {len(long_lines)} lines > 120 chars)")
        
        # Check for error handling
        if 'try:' not in content and any(risky in content for risky in ['open(', 'requests.', 'urllib', 'subprocess']):
            suggestions.append("Consider adding error handling for potentially failing operations")
        
        # Check for documentation
        if file_path.endswith('.py'):
            if '"""' not in content and "'''" not in content:
                suggestions.append("Consider adding docstrings for better documentation")
        
        return suggestions

# ----------------------------
# ENHANCED DATABASE MANAGER
# ----------------------------
class EnhancedDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            # Original tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    path TEXT PRIMARY KEY,
                    size INTEGER,
                    modified_time REAL,
                    hash TEXT,
                    language TEXT,
                    complexity_score REAL DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0,
                    last_analyzed TIMESTAMP,
                    technologies TEXT DEFAULT '[]'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    timestamp TIMESTAMP,
                    suggestions TEXT,
                    errors TEXT,
                    warnings TEXT,
                    complexity_metrics TEXT,
                    technologies_detected TEXT,
                    confidence_score REAL,
                    FOREIGN KEY (file_path) REFERENCES file_metadata (path)
                )
            """)
            
            # New telecom-specific tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS network_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    technology TEXT,
                    timestamp TIMESTAMP,
                    protocol_violations TEXT,
                    performance_metrics TEXT,
                    optimization_suggestions TEXT,
                    complexity_score REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS physics_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    calculation_type TEXT,
                    timestamp TIMESTAMP,
                    input_parameters TEXT,
                    results TEXT,
                    units TEXT,
                    validity_range TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_io TEXT,
                    processes_monitored INTEGER
                )
            """)
            
            conn.commit()
    
    def save_network_analysis(self, analysis: NetworkAnalysis):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO network_analysis 
                (file_path, technology, timestamp, protocol_violations, 
                 performance_metrics, optimization_suggestions, complexity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "", analysis.technology.value, analysis.timestamp,
                json.dumps(analysis.protocol_violations),
                json.dumps(analysis.performance_metrics),
                json.dumps(analysis.optimization_suggestions),
                analysis.complexity_score
            ))
            conn.commit()
    
    def save_physics_calculation(self, calc: PhysicsCalculation):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO physics_calculations
                (file_path, calculation_type, timestamp, input_parameters,
                 results, units, validity_range)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "", calc.calculation_type, calc.timestamp,
                json.dumps(calc.input_parameters),
                json.dumps(calc.results, default=str),
                json.dumps(calc.units),
                json.dumps(calc.validity_range, default=str)
            ))
            conn.commit()

# ----------------------------
# PHYSICS CALCULATION ENGINE
# ----------------------------
class PhysicsEngine:
    """Advanced physics calculations for telecom applications"""
    
    def __init__(self):
        self.constants = {
            'c': 299792458,  # Speed of light (m/s)
            'k': 1.38064852e-23,  # Boltzmann constant (J/K)
            'h': 6.62607004e-34,  # Planck constant (J·s)
            'e': 1.602176634e-19,  # Elementary charge (C)
            'epsilon_0': 8.8541878128e-12,  # Vacuum permittivity (F/m)
            'mu_0': 4 * math.pi * 1e-7,  # Vacuum permeability (H/m)
        }
    
    def calculate_free_space_path_loss(self, frequency_hz: float, distance_m: float) -> Dict[str, Any]:
        """Calculate free space path loss"""
        try:
            # FSPL (dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
            wavelength = self.constants['c'] / frequency_hz
            fspl_linear = (4 * math.pi * distance_m / wavelength) ** 2
            fspl_db = 10 * math.log10(fspl_linear)
            
            return {
                'fspl_db': fspl_db,
                'fspl_linear': fspl_linear,
                'wavelength_m': wavelength,
                'frequency_ghz': frequency_hz / 1e9,
                'distance_km': distance_m / 1000
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_thermal_noise(self, bandwidth_hz: float, temperature_k: float = 290) -> Dict[str, Any]:
        """Calculate thermal noise power"""
        try:
            # N = k * T * B (W)
            noise_power_w = self.constants['k'] * temperature_k * bandwidth_hz
            noise_power_dbm = 10 * math.log10(noise_power_w * 1000)  # Convert to mW then dBm
            noise_power_dbw = 10 * math.log10(noise_power_w)
            
            return {
                'noise_power_w': noise_power_w,
                'noise_power_dbm': noise_power_dbm,
                'noise_power_dbw': noise_power_dbw,
                'bandwidth_mhz': bandwidth_hz / 1e6,
                'temperature_k': temperature_k
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_shannon_capacity(self, bandwidth_hz: float, snr_db: float) -> Dict[str, Any]:
        """Calculate Shannon channel capacity"""
        try:
            snr_linear = 10 ** (snr_db / 10)
            capacity_bps = bandwidth_hz * math.log2(1 + snr_linear)
            capacity_mbps = capacity_bps / 1e6
            spectral_efficiency = capacity_bps / bandwidth_hz
            
            return {
                'capacity_bps': capacity_bps,
                'capacity_mbps': capacity_mbps,
                'spectral_efficiency_bps_hz': spectral_efficiency,
                'snr_db': snr_db,
                'snr_linear': snr_linear,
                'bandwidth_mhz': bandwidth_hz / 1e6
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_antenna_gain(self, efficiency: float, diameter_m: float, frequency_hz: float) -> Dict[str, Any]:
        """Calculate parabolic antenna gain"""
        try:
            wavelength = self.constants['c'] / frequency_hz
            area = math.pi * (diameter_m / 2) ** 2
            gain_linear = efficiency * (4 * math.pi * area) / (wavelength ** 2)
            gain_db = 10 * math.log10(gain_linear)
            gain_dbi = gain_db  # Assuming isotropic reference
            
            return {
                'gain_db': gain_db,
                'gain_dbi': gain_dbi,
                'gain_linear': gain_linear,
                'efficiency': efficiency,
                'diameter_m': diameter_m,
                'area_m2': area,
                'wavelength_m': wavelength,
                'frequency_ghz': frequency_hz / 1e9
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_fade_margin(self, availability_percent: float, path_length_km: float, 
                            frequency_ghz: float, rain_region: str = 'temperate') -> Dict[str, Any]:
        """Calculate rain fade margin"""
        try:
            # ITU-R P.838 rain attenuation model (simplified)
            rain_rates = {'tropical': 150, 'temperate': 42, 'dry': 12}  # mm/h for 0.01% time
            rain_rate = rain_rates.get(rain_region, 42)
            
            # Rain attenuation coefficient (simplified)
            if frequency_ghz < 10:
                k, alpha = 0.0001, 1.0
            elif frequency_ghz < 20:
                k, alpha = 0.01, 1.2
            else:
                k, alpha = 0.1, 1.3
            
            specific_attenuation = k * (rain_rate ** alpha)  # dB/km
            path_attenuation = specific_attenuation * path_length_km
            
            # Fade margin for desired availability
            unavailability = 100 - availability_percent
            fade_margin = path_attenuation * (unavailability / 0.01)  # Scale from 0.01% reference
            
            return {
                'fade_margin_db': fade_margin,
                'specific_attenuation_db_km': specific_attenuation,
                'path_attenuation_db': path_attenuation,
                'rain_rate_mm_h': rain_rate,
                'availability_percent': availability_percent,
                'path_length_km': path_length_km,
                'frequency_ghz': frequency_ghz
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_eirp(self, tx_power_dbm: float, antenna_gain_dbi: float, 
                      cable_loss_db: float = 0) -> Dict[str, Any]:
        """Calculate Effective Isotropic Radiated Power"""
        try:
            eirp_dbm = tx_power_dbm + antenna_gain_dbi - cable_loss_db
            eirp_w = 10 ** ((eirp_dbm - 30) / 10)  # Convert dBm to W
            
            return {
                'eirp_dbm': eirp_dbm,
                'eirp_w': eirp_w,
                'tx_power_dbm': tx_power_dbm,
                'antenna_gain_dbi': antenna_gain_dbi,
                'cable_loss_db': cable_loss_db
            }
        except Exception as e:
            return {'error': str(e)}

# ----------------------------
# SYSTEM MONITOR
# ----------------------------
class SystemMonitor:
    """Monitor system resources for optimal performance"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.config.monitor_system_resources:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()
                self._log_metrics(metrics)
                
                # Check for resource issues
                if metrics['cpu_percent'] > 90:
                    logger.warning(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
                
                if metrics['memory_percent'] > 90:
                    logger.warning(f"High memory usage: {metrics['memory_percent']:.1f}%")
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU usage
            with open('/proc/loadavg', 'r') as f:
                load_avg = f.read().split()
                cpu_load_1min = float(load_avg[0])
            
            # Memory usage
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.readlines()
            
            memory_info = {}
            for line in meminfo:
                key, value = line.split(':')
                memory_info[key] = int(value.split()[0]) * 1024  # Convert to bytes
            
            total_mem = memory_info['MemTotal']
            available_mem = memory_info.get('MemAvailable', memory_info['MemFree'])
            used_mem = total_mem - available_mem
            
            # Disk usage
            disk_usage = shutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now(),
                'cpu_load_1min': cpu_load_1min,
                'cpu_percent': min(cpu_load_1min * 100 / os.cpu_count(), 100),
                'memory_total_gb': total_mem / (1024**3),
                'memory_used_gb': used_mem / (1024**3),
                'memory_percent': (used_mem / total_mem) * 100,
                'disk_total_gb': disk_usage.total / (1024**3),
                'disk_used_gb': disk_usage.used / (1024**3),
                'disk_percent': (disk_usage.used / disk_usage.total) * 100,
                'processes': len(os.listdir('/proc')) if os.path.exists('/proc') else 0
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to console and database"""
        if 'error' in metrics:
            return
        
        if RICH_AVAILABLE:
            # Only log significant changes or every 10 minutes
            pass  # Could implement detailed logging here
        else:
            logger.info(f"CPU: {metrics['cpu_percent']:.1f}%, "
                       f"Memory: {metrics['memory_percent']:.1f}%, "
                       f"Disk: {metrics['disk_percent']:.1f}%")

# ----------------------------
# KAFKA INTEGRATION
# ----------------------------
class KafkaManager:
    """Manage Kafka operations for telecom data streaming"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.telecom_config = config.telecom_config
        self.mock_mode = True  # Always use mock mode for offline operation
        
        if RICH_AVAILABLE:
            console.print("[yellow]Kafka: Running in offline mock mode[/yellow]")
    
    def analyze_kafka_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Kafka-related code"""
        suggestions = []
        warnings = []
        errors = []
        
        # Check for producer configurations
        if "KafkaProducer" in content:
            if "acks" not in content:
                suggestions.append("Consider setting 'acks' parameter for delivery guarantees")
            if "retries" not in content:
                suggestions.append("Add retry configuration for fault tolerance")
            if "batch.size" not in content and "batch_size" not in content:
                suggestions.append("Configure batching for better throughput")
        
        # Check for consumer configurations
        if "KafkaConsumer" in content:
            if "group.id" not in content and "group_id" not in content:
                errors.append("Consumer group ID is required")
            if "enable.auto.commit" not in content:
                suggestions.append("Consider explicit offset commit strategy")
        
        # Topic naming conventions
        topic_matches = re.findall(r'["\']([a-zA-Z0-9\-_.]+)["\']', content)
        for topic in topic_matches:
            if topic in self.telecom_config.kafka_topics:
                continue
            if not re.match(r'^[a-zA-Z0-9\-_.]+$', topic):
                warnings.append(f"Topic name '{topic}' may not follow naming conventions")
        
        return {
            'suggestions': suggestions,
            'warnings': warnings,
            'errors': errors,
            'telecom_topics_found': [t for t in topic_matches if t in self.telecom_config.kafka_topics]
        }
    
    def simulate_telecom_data_flow(self) -> Dict[str, Any]:
        """Simulate telecom data patterns for analysis"""
        # Generate mock telecom metrics
        cell_metrics = {
            'cell_id': 'eNB_001_Cell_1',
            'rsrp_dbm': -85.5,
            'rsrq_db': -12.3,
            'sinr_db': 15.2,
            'throughput_mbps': 45.8,
            'active_users': 23,
            'timestamp': datetime.now().isoformat()
        }
        
        ue_event = {
            'event_type': 'handover_success',
            'ue_id': 'IMSI_123456789012345',
            'source_cell': 'eNB_001_Cell_1',
            'target_cell': 'eNB_002_Cell_3',
            'ho_duration_ms': 45,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'cell_metrics': cell_metrics,
            'ue_events': [ue_event],
            'kpis': {
                'call_success_rate': 99.2,
                'handover_success_rate': 98.7,
                'average_throughput': 42.5
            }
        }

# ----------------------------
# ENHANCED FILE HANDLER
# ----------------------------
class TelecomCodeHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    def __init__(self, config: EnhancedConfig, db_manager: EnhancedDatabaseManager,
                 analysis_engine: OfflineAnalysisEngine, physics_engine: PhysicsEngine):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self.config = config
        self.db_manager = db_manager
        self.analysis_engine = analysis_engine
        self.physics_engine = physics_engine
        self.kafka_manager = KafkaManager(config)
        self.processing_queue = queue.Queue()
        self.recent_changes = defaultdict(float)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
    
    def on_modified(self, event):
        if not event.is_directory and WATCHDOG_AVAILABLE:
            self._handle_file_change(event.src_path, "modified")
    
    def on_created(self, event):
        if not event.is_directory and WATCHDOG_AVAILABLE:
            self._handle_file_change(event.src_path, "created")
    
    def _should_process_file(self, file_path: str) -> bool:
        """Enhanced file filtering for telecom development"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in self.config.extensions:
            return False
        
        # Check ignore patterns
        for pattern in self.config.ignore_patterns:
            if pattern.replace("*", "") in str(path):
                return False
        
        # Check file size
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                return False
        except OSError:
            return False
        
        # Debouncing
        current_time = time.time()
        if current_time - self.recent_changes[file_path] < 2:
            return False
        self.recent_changes[file_path] = current_time
        
        return True
    
    def _handle_file_change(self, file_path: str, change_type: str):
        """Handle file system events"""
        if self._should_process_file(file_path):
            self.processing_queue.put((file_path, change_type))
    
    def _process_queue(self):
        """Process file changes in background"""
        while True:
            try:
                file_path, change_type = self.processing_queue.get(timeout=1)
                self._analyze_file(file_path, change_type)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    def _analyze_file(self, file_path: str, change_type: str):
        """Enhanced file analysis with telecom focus"""
        try:
            if RICH_AVAILABLE:
                console.print(f"\n[cyan]🔍 Analyzing {change_type}: {Path(file_path).name}[/cyan]")
            else:
                print(f"Analyzing {change_type}: {file_path}")
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return
            
            # Generate file metadata
            file_stats = Path(file_path).stat()
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Perform analysis
            analysis_results = self.analysis_engine.analyze_code(content, file_path)
            
            # Additional telecom-specific analysis
            if any(tech.value in str(analysis_results.get('technologies_detected', [])) 
                   for tech in [TelecomTechnology.KAFKA]):
                kafka_analysis = self.kafka_manager.analyze_kafka_code(content, file_path)
                analysis_results['kafka_analysis'] = kafka_analysis
            
            # Physics calculations if applicable
            if TelecomTechnology.COMPLEX_PHYSICS in analysis_results.get('technologies_detected', []):
                physics_analysis = self._perform_physics_analysis(content)
                analysis_results['physics_analysis'] = physics_analysis
            
            # Display results
            self._display_enhanced_results(file_path, analysis_results)
            
        except Exception as e:
            error_msg = f"Analysis failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            if RICH_AVAILABLE:
                console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"ERROR: {error_msg}")
    
    def _perform_physics_analysis(self, content: str) -> Dict[str, Any]:
        """Perform physics calculations found in code"""
        results = {}
        
        # Look for frequency calculations
        freq_matches = re.findall(r'(\d+\.?\d*)\s*(?:hz|mhz|ghz)', content, re.IGNORECASE)
        if freq_matches:
            for freq_str in freq_matches[:3]:  # Limit to first 3
                try:
                    freq = float(freq_str)
                    # Assume MHz if reasonable range
                    if 1 < freq < 10000:
                        freq_hz = freq * 1e6
                        # Calculate some basic properties
                        wavelength = self.physics_engine.constants['c'] / freq_hz
                        results[f'frequency_{freq}MHz'] = {
                            'wavelength_m': wavelength,
                            'wavelength_cm': wavelength * 100
                        }
                except ValueError:
                    continue
        
        return results
    
    def _display_enhanced_results(self, file_path: str, results: Dict[str, Any]):
        """Display enhanced analysis results"""
        if not RICH_AVAILABLE:
            # Fallback to basic display
            print(f"\nAnalysis Results for {Path(file_path).name}")
            print(f"Technologies: {results.get('technologies_detected', [])}")
            print(f"Suggestions: {len(results.get('suggestions', []))}")
            print(f"Warnings: {len(results.get('warnings', []))}")
            print(f"Errors: {len(results.get('errors', []))}")
            return
        
        # Rich display
        panel_content = []
        
        # File info
        panel_content.append(f"[bold blue]File:[/bold blue] {Path(file_path).name}")
        panel_content.append(f"[dim]Path: {file_path}[/dim]")
        panel_content.append(f"[dim]Analyzed: {datetime.now().strftime('%H:%M:%S')}[/dim]")
        panel_content.append("")
        
        # Technologies detected
        if results.get('technologies_detected'):
            panel_content.append("[bold cyan]🔧 Technologies Detected:[/bold cyan]")
            for tech in results['technologies_detected']:
                panel_content.append(f"  • {tech.value if hasattr(tech, 'value') else tech}")
            panel_content.append("")
        
        # Complexity metrics
        if results.get('complexity_metrics'):
            panel_content.append("[bold yellow]📊 Complexity Metrics:[/bold yellow]")
            for key, value in results['complexity_metrics'].items():
                panel_content.append(f"  {key}: {value}")
            panel_content.append("")
        
        # Errors
        if results.get('errors'):
            panel_content.append("[bold red]❌ Errors:[/bold red]")
            for error in results['errors']:
                panel_content.append(f"  • {error}")
            panel_content.append("")
        
        # Warnings
        if results.get('warnings'):
            panel_content.append("[bold orange3]⚠️  Warnings:[/bold orange3]")
            for warning in results['warnings'][:5]:
                panel_content.append(f"  • {warning}")
            panel_content.append("")
        
        # Suggestions
        if results.get('suggestions'):
            panel_content.append("[bold green]💡 Suggestions:[/bold green]")
            for i, suggestion in enumerate(results['suggestions'][:5], 1):
                panel_content.append(f"  {i}. {suggestion}")
            panel_content.append("")
        
        # Kafka analysis
        if results.get('kafka_analysis'):
            kafka = results['kafka_analysis']
            if kafka.get('telecom_topics_found'):
                panel_content.append("[bold magenta]📡 Telecom Kafka Topics:[/bold magenta]")
                for topic in kafka['telecom_topics_found']:
                    panel_content.append(f"  • {topic}")
                panel_content.append("")
        
        # Physics analysis
        if results.get('physics_analysis'):
            panel_content.append("[bold cyan]🧮 Physics Calculations:[/bold cyan]")
            for calc, data in results['physics_analysis'].items():
                panel_content.append(f"  • {calc}: {data}")
            panel_content.append("")
        
        # Display panel
        console.print(Panel(
            "\n".join(panel_content),
            title="🤖 Enhanced Telecom Analysis",
            border_style="bright_blue"
        ))

# ----------------------------
# MAIN ENHANCED AI SIDEKICK
# ----------------------------
class EnhancedTelecomSidekick:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.db_manager = EnhancedDatabaseManager(self.config.db_path)
        self.analysis_engine = OfflineAnalysisEngine(self.config)
        self.physics_engine = PhysicsEngine()
        self.system_monitor = SystemMonitor(self.config)
        self.observer = Observer() if WATCHDOG_AVAILABLE else None
        self.handler = None
        self.running = False
        self.stats = {
            'files_analyzed': 0,
            'technologies_detected': defaultdict(int),
            'errors_detected': 0,
            'suggestions_made': 0,
            'start_time': None
        }
    
    def _load_config(self, config_path: Optional[str]) -> EnhancedConfig:
        """Load enhanced configuration"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return EnhancedConfig(**config_data)
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]Warning: Failed to load config from {config_path}: {e}[/yellow]")
                    console.print("[yellow]Using default configuration[/yellow]")
                else:
                    print(f"Warning: Failed to load config, using defaults: {e}")
        
        return EnhancedConfig()
    
    def save_config(self, config_path: str):
        """Save current configuration"""
        try:
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            if RICH_AVAILABLE:
                console.print(f"[green]Configuration saved to {config_path}[/green]")
            else:
                print(f"Configuration saved to {config_path}")
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Failed to save config: {e}[/red]")
            else:
                print(f"Failed to save config: {e}")
    
    def setup_watchers(self):
        """Setup enhanced file system watchers"""
        if not WATCHDOG_AVAILABLE:
            if RICH_AVAILABLE:
                console.print("[yellow]File watching disabled - watchdog not available[/yellow]")
            else:
                print("Warning: File watching disabled")
            return
        
        self.handler = TelecomCodeHandler(
            self.config, self.db_manager, self.analysis_engine, self.physics_engine
        )
        
        for watch_path in self.config.watch_paths:
            if Path(watch_path).exists():
                self.observer.schedule(self.handler, path=watch_path, recursive=True)
                if RICH_AVAILABLE:
                    console.print(f"[blue]👀 Watching: {watch_path}[/blue]")
                else:
                    print(f"Watching: {watch_path}")
            else:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]⚠️  Warning: Path does not exist: {watch_path}[/yellow]")
                else:
                    print(f"Warning: Path does not exist: {watch_path}")
    
    def start(self):
        """Start the enhanced AI sidekick"""
        if RICH_AVAILABLE:
            console.print("\n" + "="*80)
            console.print("[bold green]🤖 Enhanced Telecom AI Coding Sidekick Starting...[/bold green]")
            console.print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("Enhanced Telecom AI Coding Sidekick Starting...")
            print("="*80 + "\n")
        
        self.stats['start_time'] = datetime.now()
        
        # Display startup information
        self._display_startup_info()
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Setup and start watchers
        self.setup_watchers()
        if WATCHDOG_AVAILABLE and self.observer:
            self.observer.start()
        
        self.running = True
        
        if RICH_AVAILABLE:
            console.print("\n[bold green]✅ Enhanced AI Sidekick is now active![/bold green]")
            console.print("[dim]Press Ctrl+C to stop, or type 'help' for commands[/dim]\n")
        else:
            print("\nEnhanced AI Sidekick is now active!")
            print("Press Ctrl+C to stop, or type 'help' for commands\n")
        
        try:
            # Interactive command loop
            self._run_interactive_loop()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the enhanced AI sidekick"""
        if RICH_AVAILABLE:
            console.print("\n[bold red]🛑 Shutting down Enhanced AI Sidekick...[/bold red]")
        else:
            print("\nShutting down Enhanced AI Sidekick...")
        
        self.running = False
        self.system_monitor.stop_monitoring()
        
        if WATCHDOG_AVAILABLE and self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Display final stats
        self._display_final_stats()
        
        if RICH_AVAILABLE:
            console.print("[bold green]👋 Enhanced AI Sidekick stopped successfully![/bold green]")
        else:
            print("Enhanced AI Sidekick stopped successfully!")
    
    def _display_startup_info(self):
        """Display enhanced startup information"""
        if not RICH_AVAILABLE:
            # Basic startup info
            print("System Information:")
            print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            print(f"NumPy Available: {'Yes' if NUMPY_AVAILABLE else 'No'}")
            print(f"SciPy Available: {'Yes' if SCIPY_AVAILABLE else 'No'}")
            print(f"Offline Mode: {'Yes' if self.config.offline_mode else 'No'}")
            print(f"Ubuntu Version: {self.config.ubuntu_version}")
            print()
            return
        
        # Rich display
        # System info table
        info_table = Table(title="🖥️  Enhanced System Information", border_style="blue")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        info_table.add_row("Ubuntu Version", self.config.ubuntu_version)
        info_table.add_row("NumPy Available", "✅" if NUMPY_AVAILABLE else "❌")
        info_table.add_row("SciPy Available", "✅" if SCIPY_AVAILABLE else "❌")
        info_table.add_row("YAML Support", "✅" if YAML_AVAILABLE else "❌")
        info_table.add_row("File Watching", "✅" if WATCHDOG_AVAILABLE else "❌")
        info_table.add_row("Offline Mode", "✅" if self.config.offline_mode else "❌")
        info_table.add_row("Database", self.config.db_path)
        info_table.add_row("System Monitoring", "✅" if self.config.monitor_system_resources else "❌")
        
        console.print(info_table)
        console.print()
        
        # Telecom technologies table
        telecom_table = Table(title="📡 Supported Telecom Technologies", border_style="magenta")
        telecom_table.add_column("Technology", style="cyan")
        telecom_table.add_column("Status", style="green")
        telecom_table.add_column("Key Features", style="yellow")
        
        tech_features = {
            "LTE": "eNB, UE, EPC, RRC analysis",
            "5G NR": "gNB, 5GC, AMF, SMF analysis",
            "O-RAN": "O-RU, O-DU, RIC, xApp support",
            "srsRAN": "Configuration validation",
            "Kafka": "Producer/Consumer analysis",
            "GNU Radio": "SDR signal processing",
            "Kubernetes": "Container orchestration",
            "SSH/Moshell": "Network management",
            "Complex Physics": "RF calculations, FSPL"
        }
        
        for tech, features in tech_features.items():
            telecom_table.add_row(tech, "✅ Enabled", features)
        
        console.print(telecom_table)
        console.print()
        
        # Watch paths
        if self.config.watch_paths:
            watch_table = Table(title="📁 Watch Paths", border_style="green")
            watch_table.add_column("Path", style="cyan")
            watch_table.add_column("Status", style="green")
            watch_table.add_column("Relevant Files", style="yellow")
            
            for path in self.config.watch_paths:
                if Path(path).exists():
                    # Count relevant files
                    file_count = 0
                    try:
                        for ext in self.config.extensions[:10]:  # Sample extensions
                            file_count += len(list(Path(path).rglob(f"*{ext}")))
                            if file_count > 100:  # Cap the count for performance
                                file_count = "100+"
                                break
                    except:
                        file_count = "?"
                    
                    watch_table.add_row(path, "✅ Active", str(file_count))
                else:
                    watch_table.add_row(path, "❌ Not found", "0")
            
            console.print(watch_table)
            console.print()
        
        # Extensions summary
        ext_display = ", ".join(self.config.extensions[:15])
        if len(self.config.extensions) > 15:
            ext_display += f" ... and {len(self.config.extensions) - 15} more"
        console.print(f"[cyan]📄 Monitored Extensions:[/cyan] {ext_display}")
        console.print()
    
    def _run_interactive_loop(self):
        """Enhanced interactive command loop"""
        commands = {
            'help': self._show_help,
            'stats': self._show_stats,
            'config': self._show_config,
            'analyze': self._manual_analyze,
            'recent': self._show_recent_analyses,
            'physics': self._physics_calculator,
            'kafka': self._kafka_simulation,
            'system': self._show_system_status,
            'telecom': self._show_telecom_info,
            'clear': lambda: console.clear() if RICH_AVAILABLE else os.system('clear'),
            'quit': lambda: None,
            'exit': lambda: None
        }
        
        while self.running:
            try:
                if RICH_AVAILABLE:
                    command = Prompt.ask(
                        "\n[bold cyan]Enhanced Telecom AI Sidekick[/bold cyan]",
                        default="help"
                    ).lower().strip()
                else:
                    command = input("\nEnhanced Telecom AI Sidekick> ").lower().strip()
                    if not command:
                        command = "help"
                
                if command in ['quit', 'exit']:
                    break
                elif command in commands:
                    commands[command]()
                else:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        console.print("Type 'help' for available commands")
                    else:
                        print(f"Unknown command: {command}")
                        print("Type 'help' for available commands")
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def _show_help(self):
        """Display enhanced help information"""
        if not RICH_AVAILABLE:
            print("\nAvailable Commands:")
            print("help        - Show this help message")
            print("stats       - Display analysis statistics")
            print("config      - Show current configuration")
            print("analyze     - Manually analyze a specific file")
            print("recent      - Show recent analysis results")
            print("physics     - Physics calculator for telecom")
            print("kafka       - Kafka simulation and analysis")
            print("system      - Show system status")
            print("telecom     - Show telecom-specific information")
            print("clear       - Clear the terminal screen")
            print("quit/exit   - Stop the AI sidekick")
            return
        
        help_table = Table(title="🆘 Enhanced Commands", border_style="blue")
        help_table.add_column("Command", style="cyan", width=15)
        help_table.add_column("Description", style="white")
        
        help_table.add_row("help", "Show this help message")
        help_table.add_row("stats", "Display enhanced analysis statistics")
        help_table.add_row("config", "Show current configuration")
        help_table.add_row("analyze <file>", "Manually analyze a specific file")
        help_table.add_row("recent", "Show recent analysis results")
        help_table.add_row("physics", "Interactive physics calculator")
        help_table.add_row("kafka", "Kafka simulation and analysis tools")
        help_table.add_row("system", "Show system resource status")
        help_table.add_row("telecom", "Show telecom-specific information")
        help_table.add_row("clear", "Clear the terminal screen")
        help_table.add_row("quit/exit", "Stop the enhanced AI sidekick")
        
        console.print(help_table)
    
    def _show_stats(self):
        """Display enhanced statistics"""
        uptime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        if not RICH_AVAILABLE:
            print(f"\nStatistics:")
            print(f"Uptime: {str(uptime).split('.')[0]}")
            print(f"Files Analyzed: {self.stats['files_analyzed']}")
            print(f"Errors Detected: {self.stats['errors_detected']}")
            print(f"Suggestions Made: {self.stats['suggestions_made']}")
            return
        
        stats_table = Table(title="📊 Enhanced Statistics", border_style="green")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Uptime", str(uptime).split('.')[0])
        stats_table.add_row("Files Analyzed", str(self.stats['files_analyzed']))
        stats_table.add_row("Errors Detected", str(self.stats['errors_detected']))
        stats_table.add_row("Suggestions Made", str(self.stats['suggestions_made']))
        
        # Technology detection stats
        if self.stats['technologies_detected']:
            stats_table.add_row("", "")  # Separator
            for tech, count in self.stats['technologies_detected'].items():
                stats_table.add_row(f"  {tech} Files", str(count))
        
        # Database stats
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM file_metadata")
                file_count = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(*) FROM code_analysis")
                analysis_count = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(*) FROM network_analysis")
                network_count = cursor.fetchone()[0]
                
                stats_table.add_row("", "")  # Separator
                stats_table.add_row("Files in Database", str(file_count))
                stats_table.add_row("Total Analyses", str(analysis_count))
                stats_table.add_row("Network Analyses", str(network_count))
        except Exception as e:
            stats_table.add_row("Database Error", str(e))
        
        console.print(stats_table)
    
    def _show_config(self):
        """Display current enhanced configuration"""
        if not RICH_AVAILABLE:
            print("\nConfiguration:")
            config_data = asdict(self.config)
            for key, value in config_data.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"{key}: {value[:3]} ... and {len(value) - 3} more")
                else:
                    print(f"{key}: {value}")
            return
        
        config_data = asdict(self.config)
        
        config_tree = Tree("⚙️  Enhanced Configuration")
        
        for key, value in config_data.items():
            if key == 'telecom_config':
                telecom_branch = config_tree.add("[cyan]telecom_config[/cyan]")
                for tk, tv in value.items():
                    if isinstance(tv, list):
                        tb = telecom_branch.add(f"[cyan]{tk}[/cyan]")
                        for item in tv[:3]:
                            tb.add(f"[yellow]{item}[/yellow]")
                        if len(tv) > 3:
                            tb.add(f"[dim]... and {len(tv) - 3} more[/dim]")
                    else:
                        telecom_branch.add(f"[cyan]{tk}:[/cyan] [yellow]{tv}[/yellow]")
            elif isinstance(value, list):
                branch = config_tree.add(f"[cyan]{key}[/cyan]")
                for item in value[:5]:
                    branch.add(f"[yellow]{item}[/yellow]")
                if len(value) > 5:
                    branch.add(f"[dim]... and {len(value) - 5} more[/dim]")
            else:
                config_tree.add(f"[cyan]{key}:[/cyan] [yellow]{value}[/yellow]")
        
        console.print(config_tree)
    
    def _manual_analyze(self):
        """Manual file analysis with enhanced features"""
        if RICH_AVAILABLE:
            file_path = Prompt.ask("Enter file path to analyze")
        else:
            file_path = input("Enter file path to analyze: ").strip()
        
        if not Path(file_path).exists():
            if RICH_AVAILABLE:
                console.print(f"[red]File not found: {file_path}[/red]")
            else:
                print(f"File not found: {file_path}")
            return
        
        # Force analysis
        if hasattr(self.handler, '_analyze_file'):
            self.handler._analyze_file(file_path, "manual")
        else:
            # Fallback for when watchdog is not available
            if RICH_AVAILABLE:
                console.print("[yellow]Direct analysis (file watching disabled)[/yellow]")
            else:
                print("Direct analysis (file watching disabled)")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                results = self.analysis_engine.analyze_code(content, file_path)
                
                if RICH_AVAILABLE:
                    self._display_manual_results(file_path, results)
                else:
                    print(f"\nAnalysis Results for {Path(file_path).name}")
                    print(f"Technologies: {results.get('technologies_detected', [])}")
                    print(f"Suggestions ({len(results.get('suggestions', []))}): {results.get('suggestions', [])[:3]}")
                    print(f"Warnings ({len(results.get('warnings', []))}): {results.get('warnings', [])[:3]}")
                    print(f"Errors ({len(results.get('errors', []))}): {results.get('errors', [])}")
                
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Analysis failed: {e}[/red]")
                else:
                    print(f"Analysis failed: {e}")
    
    def _display_manual_results(self, file_path: str, results: Dict[str, Any]):
        """Display manual analysis results"""
        if not RICH_AVAILABLE:
            return
        
        panel_content = []
        
        panel_content.append(f"[bold blue]Manual Analysis:[/bold blue] {Path(file_path).name}")
        panel_content.append(f"[dim]Path: {file_path}[/dim]")
        panel_content.append("")
        
        if results.get('technologies_detected'):
            panel_content.append("[bold cyan]🔧 Technologies:[/bold cyan]")
            for tech in results['technologies_detected']:
                panel_content.append(f"  • {tech.value if hasattr(tech, 'value') else tech}")
            panel_content.append("")
        
        if results.get('complexity_metrics'):
            panel_content.append("[bold yellow]📊 Metrics:[/bold yellow]")
            for key, value in results['complexity_metrics'].items():
                panel_content.append(f"  {key}: {value}")
            panel_content.append("")
        
        if results.get('suggestions'):
            panel_content.append("[bold green]💡 Top Suggestions:[/bold green]")
            for i, suggestion in enumerate(results['suggestions'][:3], 1):
                panel_content.append(f"  {i}. {suggestion}")
        
        console.print(Panel(
            "\n".join(panel_content),
            title="🔍 Manual Analysis Results",
            border_style="bright_green"
        ))
    
    def _physics_calculator(self):
        """Interactive physics calculator"""
        if not RICH_AVAILABLE:
            print("\nPhysics Calculator")
            print("Available calculations:")
            print("1. Free Space Path Loss")
            print("2. Thermal Noise")
            print("3. Shannon Capacity")
            print("4. Antenna Gain")
            print("5. EIRP")
            
            choice = input("Choose calculation (1-5): ").strip()
            
            try:
                if choice == "1":
                    freq = float(input("Frequency (Hz): "))
                    dist = float(input("Distance (m): "))
                    result = self.physics_engine.calculate_free_space_path_loss(freq, dist)
                    print(f"FSPL: {result.get('fspl_db', 'Error'):.2f} dB")
                elif choice == "2":
                    bw = float(input("Bandwidth (Hz): "))
                    result = self.physics_engine.calculate_thermal_noise(bw)
                    print(f"Noise Power: {result.get('noise_power_dbm', 'Error'):.2f} dBm")
                # Add other calculations...
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid input")
            return
        
        # Rich interface
        calc_table = Table(title="🧮 Physics Calculator", border_style="cyan")
        calc_table.add_column("Option", style="cyan")
        calc_table.add_column("Calculation", style="white")
        
        calc_table.add_row("1", "Free Space Path Loss (FSPL)")
        calc_table.add_row("2", "Thermal Noise Power")
        calc_table.add_row("3", "Shannon Channel Capacity")
        calc_table.add_row("4", "Parabolic Antenna Gain")
        calc_table.add_row("5", "Effective Isotropic Radiated Power (EIRP)")
        calc_table.add_row("6", "Rain Fade Margin")
        
        console.print(calc_table)
        
        choice = Prompt.ask("Choose calculation", choices=["1", "2", "3", "4", "5", "6"])
        
        try:
            if choice == "1":
                freq = float(Prompt.ask("Frequency (Hz)", default="28.4e9"))
                dist = float(Prompt.ask("Distance (m)", default="300"))
                result = self.physics_engine.calculate_free_space_path_loss(freq, dist)
                
                if 'error' not in result:
                    result_table = Table(title="📡 Free Space Path Loss Results", border_style="green")
                    result_table.add_column("Parameter", style="cyan")
                    result_table.add_column("Value", style="yellow")
                    result_table.add_column("Unit", style="green")
                    
                    result_table.add_row("FSPL", f"{result['fspl_db']:.2f}", "dB")
                    result_table.add_row("Wavelength", f"{result['wavelength_mm']:.4f}", "mm")
                    result_table.add_row("Frequency", f"{result['frequency_ghz']:.3f}", "GHz")
                    result_table.add_row("Distance", f"{result['distance_km']:.3f}", "km")
                    
                    console.print(result_table)
                else:
                    console.print(f"[red]Calculation error: {result['error']}[/red]")
            
            elif choice == "2":
                bw = float(Prompt.ask("Bandwidth (Hz)", default="20e6"))
                temp = float(Prompt.ask("Temperature (K)", default="290"))
                result = self.physics_engine.calculate_thermal_noise(bw, temp)
                
                if 'error' not in result:
                    result_table = Table(title="🌡️ Thermal Noise Results", border_style="green")
                    result_table.add_column("Parameter", style="cyan")
                    result_table.add_column("Value", style="yellow")
                    result_table.add_column("Unit", style="green")
                    
                    result_table.add_row("Noise Power", f"{result['noise_power_dbm']:.2f}", "dBm")
                    result_table.add_row("Noise Power", f"{result['noise_power_w']:.2e}", "W")
                    result_table.add_row("Bandwidth", f"{result['bandwidth_mhz']:.1f}", "MHz")
                    result_table.add_row("Temperature", f"{result['temperature_k']:.1f}", "K")
                    
                    console.print(result_table)
            
            # Add other calculation implementations...
            
        except ValueError as e:
            console.print(f"[red]Invalid input: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Calculation error: {e}[/red]")
    
    def _kafka_simulation(self):
        """Kafka simulation and analysis"""
        if not RICH_AVAILABLE:
            print("\nKafka Simulation")
            print("Simulating telecom data flow...")
            data = self.handler.kafka_manager.simulate_telecom_data_flow() if hasattr(self.handler, 'kafka_manager') else {}
            print(f"Generated sample data: {len(data)} data types")
            return
        
        console.print("[cyan]🔄 Simulating Telecom Data Flow...[/cyan]")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Generating telecom data...", total=None)
            
            # Simulate data generation
            if hasattr(self.handler, 'kafka_manager'):
                data = self.handler.kafka_manager.simulate_telecom_data_flow()
            else:
                data = {
                    'cell_metrics': {'cell_id': 'mock_cell', 'rsrp': -85},
                    'ue_events': [{'event': 'handover_success'}]
                }
            
            progress.update(task, description="✅ Data generation complete")
        
        # Display simulated data
        data_table = Table(title="📡 Simulated Telecom Data", border_style="magenta")
        data_table.add_column("Data Type", style="cyan")
        data_table.add_column("Sample", style="yellow")
        
        for key, value in data.items():
            if isinstance(value, dict):
                sample = f"{list(value.keys())[:3]}..."
            elif isinstance(value, list):
                sample = f"Array with {len(value)} items"
            else:
                sample = str(value)[:50]
            
            data_table.add_row(key, sample)
        
        console.print(data_table)
    
    def _show_system_status(self):
        """Show system status and resources"""
        metrics = self.system_monitor.get_system_metrics()
        
        if not RICH_AVAILABLE:
            print("\nSystem Status:")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
            else:
                print(f"CPU: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"Memory: {metrics.get('memory_percent', 0):.1f}%")
                print(f"Disk: {metrics.get('disk_percent', 0):.1f}%")
            return
        
        if 'error' in metrics:
            console.print(f"[red]System metrics error: {metrics['error']}[/red]")
            return
        
        status_table = Table(title="💻 System Status", border_style="blue")
        status_table.add_column("Resource", style="cyan")
        status_table.add_column("Usage", style="yellow")
        status_table.add_column("Details", style="green")
        
        # CPU
        cpu_color = "red" if metrics.get('cpu_percent', 0) > 80 else "yellow" if metrics.get('cpu_percent', 0) > 60 else "green"
        status_table.add_row(
            "CPU", 
            f"[{cpu_color}]{metrics.get('cpu_percent', 0):.1f}%[/{cpu_color}]",
            f"Load: {metrics.get('cpu_load_1min', 0):.2f}"
        )
        
        # Memory
        mem_color = "red" if metrics.get('memory_percent', 0) > 90 else "yellow" if metrics.get('memory_percent', 0) > 75 else "green"
        status_table.add_row(
            "Memory",
            f"[{mem_color}]{metrics.get('memory_percent', 0):.1f}%[/{mem_color}]",
            f"{metrics.get('memory_used_gb', 0):.1f}GB / {metrics.get('memory_total_gb', 0):.1f}GB"
        )
        
        # Disk
        disk_color = "red" if metrics.get('disk_percent', 0) > 95 else "yellow" if metrics.get('disk_percent', 0) > 85 else "green"
        status_table.add_row(
            "Disk",
            f"[{disk_color}]{metrics.get('disk_percent', 0):.1f}%[/{disk_color}]",
            f"{metrics.get('disk_used_gb', 0):.1f}GB / {metrics.get('disk_total_gb', 0):.1f}GB"
        )
        
        status_table.add_row("Processes", str(metrics.get('processes', 0)), "Active processes")
        
        console.print(status_table)
    
    def _show_telecom_info(self):
        """Show telecom-specific information and configuration"""
        if not RICH_AVAILABLE:
            print("\nTelecom Configuration:")
            telecom_config = self.config.telecom_config
            print(f"5G Frequency Bands: {telecom_config.frequency_bands}")
            print(f"Bandwidth Options: {telecom_config.bandwidth_mhz}")
            print(f
            
