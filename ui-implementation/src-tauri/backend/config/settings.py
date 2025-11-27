from typing import Dict, Literal, Optional, Set, List, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Project root for absolute paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- CONSTANT: THE ONE TRUE MEMORY PATH ---
# Data paths - Use AppData for production builds, local for development
#
# DEV vs PROD separation:
# - ROAMPAL_DATA_DIR env var controls app folder name (e.g., "Roampal_DEV" vs "Roampal")
# - DEV build: port 8002, data in AppData/Roampal_DEV/data
# - PROD build: port 8001, data in AppData/Roampal/data

# Priority 1: Environment variable (allows override for dev/prod separation)
ROAMPAL_DATA_DIR = os.getenv("ROAMPAL_DATA_DIR")

if ROAMPAL_DATA_DIR:
    # Explicitly set via environment variable (e.g., "Roampal_DEV" for dev build)
    if os.name == 'nt':  # Windows
        appdata = os.getenv('APPDATA', os.path.expanduser('~\\AppData\\Roaming'))
        DATA_PATH = os.path.join(appdata, ROAMPAL_DATA_DIR, 'data')
    elif sys.platform == 'darwin':  # macOS
        DATA_PATH = os.path.expanduser(f'~/Library/Application Support/{ROAMPAL_DATA_DIR}/data')
    else:  # Linux
        DATA_PATH = os.path.expanduser(f'~/.local/share/{ROAMPAL_DATA_DIR.lower()}/data')

    os.makedirs(DATA_PATH, exist_ok=True)
    if "--mcp" not in sys.argv:
        print(f"[Roampal] Using ROAMPAL_DATA_DIR={ROAMPAL_DATA_DIR}: {DATA_PATH}")

elif (PROJECT_ROOT / "ui-implementation").exists():
    # Development mode - project structure includes ui-implementation folder
    DATA_PATH = str(PROJECT_ROOT / "data")
    if "--mcp" not in sys.argv:
        print(f"[Roampal] Development mode detected, using local data: {DATA_PATH}")

else:
    # Production/bundled mode - use platform-specific user data directory (default: Roampal)
    if os.name == 'nt':  # Windows
        appdata = os.getenv('APPDATA', os.path.expanduser('~\\AppData\\Roaming'))
        DATA_PATH = os.path.join(appdata, 'Roampal', 'data')
    elif sys.platform == 'darwin':  # macOS
        DATA_PATH = os.path.expanduser('~/Library/Application Support/Roampal/data')
    else:  # Linux
        DATA_PATH = os.path.expanduser('~/.local/share/roampal/data')

    # Ensure directory exists
    os.makedirs(DATA_PATH, exist_ok=True)
    # Only log to console if not in MCP mode (MCP uses stdio for protocol)
    if "--mcp" not in sys.argv:
        print(f"[Roampal] Production mode detected, using AppData: {DATA_PATH}")

ROAMPAL_DATA_PATH = DATA_PATH

# Legacy compatibility - point to new locations
OG_DATA_PATH = DATA_PATH
OG_ARBITRARY_STORE = DATA_PATH
OG_VECTOR_STORE = str(Path(DATA_PATH) / "vector_store")

# --- Add global NEURONS_JSONL_PATH for compatibility ---
# NEURONS_JSONL_PATH = str(Path(OG_ARBITRARY_STORE) / "neurons.jsonl")

# --- Global BOOKS list (deprecated; books are now per-shard and dynamic) ---
BOOKS = []

# --- Ensure prompts and og_books exist ---
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
(Path(DATA_PATH) / "books").mkdir(parents=True, exist_ok=True)

class AppSettings(BaseSettings):
    app_name: str = "Roampal Backend"
    log_level: str = "INFO"
    model_config = SettingsConfigDict(
        env_prefix='ROAMPAL_APP_',
        extra='ignore',
        case_sensitive=False
    )

class LLMSettings(BaseSettings):
    provider: Literal["ollama", "openchat", "lmstudio"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral-nemo:12b-instruct-2407-q4_0"  # Using Mistral Nemo for better performance
    ollama_request_timeout_seconds: int = 30  # 30 seconds - fail fast for MCP (was 300)
    ollama_keep_alive_seconds: int = 120
    max_keepalive_connections: int = Field(5, description="Max keepalive connections for httpx.")
    max_connections: int = Field(10, description="Max connections for httpx.")
    openchat_format: Optional[str] = None
    default_system_prompt: Optional[str] = None
    model_config = SettingsConfigDict(
        env_prefix='ROAMPAL_LLM_',
        env_nested_delimiter='__',
        extra='ignore',
        case_sensitive=False
    )

class FragmentFileMemorySettings(BaseModel):
    base_data_path: str = OG_DATA_PATH
    goals_filename: Optional[str] = "goals.txt"
    values_filename: Optional[str] = "values.txt"
    conversation_history_subdir: Optional[str] = "conversations"
    arbitrary_store_subdir: Optional[str] = "arbitrary_store"
    class Config:
        extra = 'ignore'
    @property
    def file_memory_data_path(self) -> str:
        return self.base_data_path
    @property
    def file_goals_filename(self) -> str:
        return self.goals_filename or "goals.txt"
    @property
    def file_values_filename(self) -> str:
        return self.values_filename or "values.txt"
    @property
    def file_conversation_history_dir(self) -> str:
        return str(Path(self.base_data_path).resolve() / (self.conversation_history_subdir or "conversations"))
    @property
    def file_arbitrary_data_dir(self) -> str:
        return str(Path(self.base_data_path).resolve() / (self.arbitrary_store_subdir or "arbitrary_store"))

class OGMemorySettings(FragmentFileMemorySettings):
    adapter_type: Literal["file"] = "file"
    base_data_path: str = OG_DATA_PATH

class WebSearchSettings(BaseSettings):
    playwright_service_url: str = "http://localhost:8001"
    default_search_engine: Literal["google", "duckduckgo", "bing"] = "bing"
    request_timeout_seconds: int = 60
    default_num_results: int = Field(5, description="Default number of search results.")
    model_config = SettingsConfigDict(
        env_prefix='ROAMPAL_WEB_',
        extra='ignore',
        case_sensitive=False
    )

class PromptConfigBase(BaseModel):
    template_directory: str = str(PROMPTS_DIR)
    class Config:
        extra = 'ignore'

class OGPromptSettings(PromptConfigBase):
    default_system_prompt_name: str = "og_system_prompt"

class ScoringSettings(BaseSettings):
    strategy: Literal["basic", "none"] = "basic"
    data_path: str = str(Path(OG_ARBITRARY_STORE) / "scores.json")
    model_config = SettingsConfigDict(
        env_prefix='ROAMPAL_SCORING_',
        extra='ignore',
        case_sensitive=False
    )
    @property
    def og_neuron_store_path(self) -> str:
        return str(Path(OG_ARBITRARY_STORE) / "neuron_scores.jsonl")
    def get_scoring_path(self, fragment: Optional[str] = None) -> str:
        return self.og_neuron_store_path

class EmbeddingSettings(BaseSettings):
    provider: Literal["huggingface", "ollama"] = "ollama"
    model_name: Optional[str] = "nomic-embed-text"
    huggingface_model_name: str = "bert-base-uncased"
    align_with_llm: bool = True
    model_config = SettingsConfigDict(
        env_prefix='ROAMPAL_EMBEDDING_',
        extra='ignore',
        case_sensitive=False,
        protected_namespaces=('settings_',)
    )

class FeedbackSettings(BaseModel):
    """Simplified feedback settings - focus on essential scoring only"""
    # Core scoring weights (30% sentiment, 70% LLM evaluation as per refactor)
    sentiment_weight: float = Field(0.3, description="Weight for sentiment in composite score")
    usefulness_weight: float = Field(0.7, description="Weight for LLM evaluation in composite score")
    
    # Time decay settings
    decay_rate_per_day: float = Field(0.01, description="Score decay rate per day unused (1%)")
    min_decay_factor: float = Field(0.5, description="Minimum decay factor (50% floor)")
    
    # Thresholds
    low_usefulness_threshold: float = Field(0.3, description="Threshold for low usefulness")
    high_usefulness_threshold: float = Field(0.7, description="Threshold for high usefulness")
    positive_threshold: float = Field(0.7, description="Threshold for positive sentiment")
    negative_threshold: float = Field(0.3, description="Threshold for negative sentiment")
    min_score: float = Field(0.0, description="Minimum allowed score")
    max_score: float = Field(1.0, description="Maximum allowed score")
    
    @field_validator('*', mode='after')
    @classmethod
    def validate_ranges(cls, v):
        if isinstance(v, float) and not 0 <= v <= 1:
            raise ValueError("Must be between 0 and 1")
        return v

class ChromaDBSettings(BaseModel):
    """ChromaDB configuration settings"""
    use_server: bool = Field(True, description="Use ChromaDB server mode instead of embedded mode")
    server_host: str = Field("localhost", description="ChromaDB server hostname")
    server_port: int = Field(8003, description="ChromaDB server port")
    persistence_directory: str = Field("backend/data/chromadb_server", description="ChromaDB persistence directory for server mode")
    
class MemorySettings(BaseModel):
    prune_threshold: float = Field(-0.5)
    prune_age_days: int = Field(30)
    merge_similarity_threshold: float = Field(0.8)
    merge_batch_size: int = Field(100)
    merge_confidence_threshold: float = Field(0.7)
    decay_standard_factor: float = Field(0.999)
    decay_trusted_factor: float = Field(0.9998)
    novelty_threshold: float = Field(0.2)
    min_fragment_score: float = Field(0.1)
    hybrid_top_k_initial: int = Field(15)
    vector_db_collection_name: str = Field("loopsmith_memories")
    default_history_limit: int = Field(10)

class SchedulerSettings(BaseModel):
    sleep_cycle_interval_hours: int = Field(24)
    replay_max_fragments: int = Field(100)
    decay_cycle_interval_hours: Optional[int] = Field(None)

class PromptSettings(BaseModel):
    tool_decision_template_path: Path = Field(PROMPTS_DIR / "tool_decision.txt")
    sentiment_analysis_template_path: Path = Field(PROMPTS_DIR / "sentiment_analysis.txt")
    profile_update_detection_template_path: Path = Field(PROMPTS_DIR / "profile_update_detection.txt")
    book_extraction_template_path: Path = Field(PROMPTS_DIR / "book_extraction.txt")
    fragment_rating_template_path: Path = Field(PROMPTS_DIR / "fragment_rating.txt")
    fragment_replay_template_path: Path = Field(PROMPTS_DIR / "fragment_replay.txt")
    history_summary_template_path: Path = Field(PROMPTS_DIR / "history_summary.txt")
    merge_cluster_template_path: Path = Field(PROMPTS_DIR / "merge_cluster.txt")
    tone_inference_template_path: Optional[Path] = Field(None)
    map_chunk_template_path: Path = Field(PROMPTS_DIR / "map_chunk.txt")
    reduce_summaries_template_path: Path = Field(PROMPTS_DIR / "reduce_summaries.txt")
    extract_content_template_path: Path = Field(PROMPTS_DIR / "extract_content.txt")
    debate_prompt_template_path: Path = Field(PROMPTS_DIR / "debate_prompt.txt")
    blend_inputs_template_path: Path = Field(PROMPTS_DIR / "blend_inputs.txt")

class PathSettings(BaseModel):
    data_dir: Path = Path(DATA_PATH)
    data_storage_dir: Path = Path(DATA_PATH)

    def get_vector_db_dir(self) -> Path:
        return self.data_storage_dir / "vector_store"

    def get_pending_scores_path(self) -> Path:
        return self.data_storage_dir / "pending_fragment_scores.jsonl"

    def get_prune_archive_path(self) -> Path:
        return self.data_storage_dir / "pruned_archive.jsonl"

    def get_neurons_jsonl_path(self) -> Path:
        return self.data_storage_dir / "neuron_scores.jsonl"

    def get_memory_layers_dir(self) -> Path:
        """Get memory layers directory (deprecated)"""
        return self.data_storage_dir

    def get_quotes_jsonl_path(self) -> Path:
        return self.data_storage_dir / "memory" / "quotes.jsonl"

    def get_models_jsonl_path(self) -> Path:
        return self.data_storage_dir / "memory" / "models.jsonl"

    def get_summaries_jsonl_path(self) -> Path:
        return self.data_storage_dir / "memory" / "summaries.jsonl"

    def get_learnings_jsonl_path(self) -> Path:
        return self.data_storage_dir / "memory" / "learnings.jsonl"

    def get_conversations_jsonl_path(self) -> Path:
        return self.data_storage_dir / "memory" / "conversations.jsonl"

    def get_book_folder_path(self) -> Path:
        return self.data_storage_dir / "books"

    def get_tone_state_path(self) -> Path:
        return self.data_dir / "tone_state.json"

    def get_profile_data_dir(self) -> Path:
        return self.data_storage_dir

    def get_profile_path(self) -> Path:
        return self.data_storage_dir / "profile.json"

    def get_suggestions_path(self) -> Path:
        return self.data_storage_dir / "profile_suggestions.json"

    def get_book_content_dir(self) -> Path:
        return self.data_storage_dir / "memory"

    def get_book_registry_path(self) -> Path:
        return self.get_book_content_dir() / "book_registry.json"

    def get_neuron_cache_path(self) -> Path:
        return self.data_storage_dir / "memory" / "neuron_scores.jsonl"

    def get_knowledge_graph_path(self) -> Path:
        """Get the knowledge graph directory"""
        return self.data_storage_dir / "knowledge_graph"

    def get_og_sessions_base_dir(self) -> Path:
        return self.data_storage_dir

    def get_sessions_dir(self) -> Path:
        return self.data_storage_dir / "sessions"

    def get_corrections_jsonl_path(self) -> Path:
        return self.data_storage_dir / "corrections.jsonl"

    def get_base_project_path(self) -> Path:
        """Get the base project root path"""
        return PROJECT_ROOT
    
    def get_legacy_vector_store_path(self) -> Path:
        """Get the legacy vector store path for backward compatibility"""
        return self.data_dir / "vector_store"
    
    def get_legacy_book_content_path(self) -> Path:
        """Get the legacy book content path for backward compatibility"""
        return self.data_dir / "book content"

    prompt_template_dir: Path = Field(PROMPTS_DIR)

class ThresholdSettings(BaseModel):
    token_budget_memory_context: int = Field(1200)
    token_limit_memory_extract: int = Field(700)
    raw_window_memory_extract: int = Field(5)
    max_window_memory_extract: int = Field(20)
    code_window_memory_extract: int = Field(20)
    min_code_turns: int = Field(2)
    web_search_num_results: int = Field(3)
    hybrid_min_score_threshold: float = Field(0.1)
    cosine_dedup_threshold: float = Field(0.95)
    levenshtein_threshold: Optional[int] = Field(None)
    num_bullets_summary: int = Field(5)

class KnowledgeGraphSettings(BaseModel):
    enabled: bool = True
    max_relationships_per_concept: int = 10
    min_relationship_confidence: float = 0.7
    graph_analysis_prompt: str = "knowledge_graph_analysis.txt"
    max_graph_hops: int = 3
    graph_analysis_batch_size: int = 50


class MemoryLayerSettings(BaseModel):
    """Simplified memory layer settings - focus on core memory functionality (deprecated)"""
    # Core memory settings
    default_strength: float = 0.5  # Normalized to 0-1
    max_strength: float = 1.0      # Normalized to 0-1
    min_strength: float = 0.0
    
    # Search settings
    hybrid_top_k_initial: int = Field(15, description="Number of results for hybrid search")
    
    # Simplified scoring - just sentiment and usefulness as per refactor
    # Removed: temporal_relevance, emotional_weight, domain_specificity, life_stage_relevance
    
    # Cluster settings (for deduplication)
    cluster_similarity_threshold: float = 0.9  # Higher threshold for deduplication
    max_cluster_mates: int = 5
    
    # Seeding settings (unchanged as they work fine)
    seed_num_quotes: int = 10
    seed_num_models: int = 5
    seed_num_summaries: int = 3
    seed_num_learnings: int = 5
    seed_num_conversations: int = 3
    seeded_min_fragments: int = 5

class ToneSettings(BaseModel):
    modes: Dict[str, str] = Field({
        'build': 'precise, strategic, minimal. No fluff—focus on steps.',
        'vision': 'metaphorical, flowing, energizing. Inspire with big-picture ideas.',
        'burnout': 'calm, supportive, no pressure. Short, reassuring responses.',
        'vent': 'blunt, loyal, validating. Call it like it is, no formalities or sign-offs.',
        'default': 'grounded, balanced. Casual and clear.',
        'supportive': 'empathetic, patient, understanding. Focus on helping and encouraging.'
    })
    emotion_keywords: Dict[str, str] = Field({
        'frustrated': 'vent', 'annoying': 'vent', 'tired': 'burnout',
        'give up': 'burnout', 'meh': 'burnout', 'excited': 'vision',
        'bullshit': 'vent', 'rude': 'vent', 'exhausting': 'burnout', 'sick': 'vent', 'bs': 'vent'
    })
    polarity_negative_threshold: float = Field(-0.3)
    polarity_positive_threshold: float = Field(0.5)
    polarity_switch_threshold: float = Field(0.4)
    history_maxlen: int = Field(3)

class MemoryInjectorSettings(BaseModel):
    """Simplified context injection settings (deprecated)"""
    token_budget: int = Field(2048, description="Max tokens for context injection")
    priority_order: List[str] = Field(
        ["user", "assistant", "book_quote", "summary"],
        description="Priority order for context types"
    )
    max_items_per_type: Dict[str, int] = Field({
        "user": 3, 
        "assistant": 3, 
        "book_quote": 2, 
        "summary": 2
    }, description="Maximum items per content type")
    forgotten_score_threshold: float = Field(0.15, description="Threshold for considering a fragment forgotten")
    forgotten_revive_probability: float = Field(0.1, description="Probability of reviving a forgotten fragment")
    decay_rate_neutral: float = Field(0.02, description="Decay rate for neutral fragments")
    min_score: float = Field(0.0, description="Minimum score for fragments")
    positive_threshold: float = Field(0.6, description="Sentiment threshold for positive boost")
    negative_threshold: float = Field(-0.6, description="Sentiment threshold for negative decay")
    boost_positive: float = Field(0.15, description="Score boost for positive sentiment")
    decay_negative: float = Field(0.1, description="Score decay for negative sentiment")

class TextSettings(BaseModel):
    english_stop_words: Set[str] = Field({
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "did", "do", "does", "doing", "down", "during",
        "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "know",
        "me", "more", "most", "my", "myself",
        "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
        "s", "same", "she", "should", "so", "some", "such",
        "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves"
    })
    fluff_phrases: List[str] = Field([
        "in your opinion", "what do you think about", "can you tell me", "i'd like to know",
        "could you please explain", "tell me more about", "i'm curious about", "what are your thoughts on",
        "can we discuss", "let's talk about"
    ])

class SecretsSettings(BaseModel):
    ollama_api_key: Optional[str] = Field(None)
    huggingface_token: Optional[str] = Field(None)

class LoggingSettings(BaseModel):
    level: str = Field("INFO")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[Path] = Field(None)

class PerformanceSettings(BaseModel):
    book_processor_concurrency: int = Field(1)
    async_semaphore_limit: int = Field(5)

class FeatureFlags(BaseModel):
    enable_self_debate: bool = Field(True)
    enable_web_fallback: bool = Field(True)
    disable_seeding: bool = Field(False, description="Disable book seeding on startup to improve performance")

class Settings(BaseSettings):
    active_shard: str = Field("roampal", description="The tool name.")
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    og_memory: OGMemorySettings = Field(default_factory=OGMemorySettings)
    fragment_memory_configs: Dict[str, FragmentFileMemorySettings] = Field(default_factory=dict)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    og_prompt_config: OGPromptSettings = Field(default_factory=OGPromptSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    processor_version: str = "1.0"
    feedback: FeedbackSettings = FeedbackSettings()
    memory: MemorySettings = MemorySettings()
    chromadb: ChromaDBSettings = ChromaDBSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    prompts: PromptSettings = PromptSettings()
    paths: PathSettings = PathSettings()
    thresholds: ThresholdSettings = ThresholdSettings()
    memory_layer: MemoryLayerSettings = MemoryLayerSettings()
    knowledge_graph: KnowledgeGraphSettings = KnowledgeGraphSettings()
    tone: ToneSettings = ToneSettings()
    memory_injector: MemoryInjectorSettings = MemoryInjectorSettings()
    text: TextSettings = TextSettings()
    secrets: SecretsSettings = SecretsSettings()
    logging: LoggingSettings = LoggingSettings()
    performance: PerformanceSettings = PerformanceSettings()
    flags: FeatureFlags = FeatureFlags()
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / '.env'),
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
        env_nested_delimiter='__',
        env_prefix = "OG_"
    )
    def __post_init__(self):
        if self.embedding.align_with_llm and self.embedding.provider == "ollama":
            self.embedding.model_name = self.embedding.model_name or self.llm.ollama_model
            logger.info("Aligned embedding model with active LLM: {}".format(self.embedding.model_name))

    def set_active_shard(self, shard_id: str):
        """Method to switch the active shard (call from UI/backend on switch)."""
        self.active_shard = shard_id
        os.environ["ACTIVE_SHARD"] = shard_id  # Persist to env for subprocesses
        logger.info(f"Active shard switched to '{shard_id}'.")

settings = Settings()

def load_additional_fragment_configs(settings_instance: Settings, config_file_name: str = "fragments_memory_config.json"):
    project_root = PROJECT_ROOT
    config_file_path = project_root / "config" / config_file_name
    if config_file_path.exists():
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                fragment_configs_dict = json.load(f)
                for frag_id, config_data in fragment_configs_dict.items():
                    if "base_data_path" in config_data:
                        config_data["base_data_path"] = OG_DATA_PATH
                        settings_instance.fragment_memory_configs[frag_id] = FragmentFileMemorySettings(**config_data)
                        logger.info(f"Loaded memory config for fragment: '{frag_id}' from {config_file_path} using base_data_path: {OG_DATA_PATH}")
                    else:
                        logger.warning(f"Skipping fragment config for '{frag_id}' due to missing 'base_data_path' in {config_file_path}")
        except Exception as e:
            logger.error(f"Error loading fragment memory configurations from {config_file_path}: {e}", exc_info=True)
    else:
        logger.info(f"Fragment memory configuration file '{config_file_name}' not found at {config_file_path}. No additional fragments loaded by this function.")

load_additional_fragment_configs(settings)
