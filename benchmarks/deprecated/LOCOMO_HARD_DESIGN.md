# LOCOMO-HARD: Testing Learning in Conversational Memory

## The Problem with Standard LOCOMO

**Standard LOCOMO**: 10 conversations, ask questions about each conversation
- **Weakness**: All info is relevant → simple vector search dominates
- **Doesn't test**: Learning, noise filtering, source prioritization
- **Result**: SimpleMemoryStore gets 100%, learning systems don't shine

## LOCOMO-HARD: What Makes It Actually Hard

### Enhancement 1: **Noise Injection** (Semantic Confusion)
For each conversation, inject 4× NOISE that's semantically similar but factually different:

**Example:**
- **Real conversation**: John and Maria discuss John's yoga class with Rob
- **Noise injected**:
  - John and Sarah discuss John's yoga class with Mike
  - John and Lisa discuss John's meditation class with Rob
  - Tom and Maria discuss Tom's yoga class with Rob
  - John and Maria discuss John's gym class with Rob

**All noise**:
- Uses same names (but swapped)
- Uses same topics (yoga, classes, activities)
- Semantically VERY similar to real conversation
- But factually WRONG

**Test**: Ask "Who did John go to yoga with?"
- **SimpleMemoryStore**: Retrieves all 5 similar texts → Confused → 50% accuracy
- **Learning system with outcomes**: Gets "Rob" wrong first, user feedback "worked", learns to prioritize real conversation → 80%+ accuracy

---

### Enhancement 2: **Temporal Contradictions** (Outcome-Based Correction)
Inject OLD/OUTDATED information that was once true but changed:

**Example:**
- **Turn 1** (June): "John goes to Yoga Studio A with Rob"
- **Turn 50** (August): "John switched to Yoga Studio B with Sarah"
- **Turn 100** (October): "John quit yoga, now does pilates with Rob"

**Test**: Ask "Where does John do yoga now?"
- **SimpleMemoryStore**: Retrieves all 3 → Confused → Likely picks earlier (more similar) answer → WRONG
- **Learning system**:
  - First attempt: Might say "Studio A" (old info)
  - User feedback: "That's outdated" → outcome=failed
  - System learns: Downvote old memories, prioritize recent context
  - Next attempt: "John doesn't do yoga anymore, he does pilates" → CORRECT

---

### Enhancement 3: **Cross-Conversation Interference**
Multiple conversations with SIMILAR names/topics that conflict:

**Setup:**
- **Conv 1**: John (software engineer) and Maria discuss Python coding
- **Conv 2**: John (chef) and Maria discuss Python (the snake) at zoo
- **Conv 3**: John (student) and Maria discuss Python (Monty Python movies)

**Test**: Ask "What does John know about Python?"
- **SimpleMemoryStore**: Retrieves all 3 → Mixes them together → "John codes snakes while watching movies?" → WRONG
- **Learning system with KG routing**:
  - Learns conv_1 answers coding questions (high success rate)
  - Learns conv_2 answers animal questions
  - Learns conv_3 answers entertainment questions
  - Routes correctly based on learned patterns → 90%+ accuracy

---

### Enhancement 4: **Progressive Difficulty** (Learning Curve)
Ask questions in PHASES with outcome feedback:

**Phase 1** (Q1-20): Cold start, no learned patterns
- Expected: 60% accuracy (lots of noise)

**Phase 2** (Q21-50): After 20 outcomes recorded
- Expected: 75% accuracy (starting to learn which sources work)

**Phase 3** (Q51-100): After 50 outcomes
- Expected: 85%+ accuracy (strong learned patterns)

**Phase 4** (Q101-150): Novel questions testing generalization
- Expected: 80%+ accuracy (patterns transfer to new queries)

**Metrics:**
- **SimpleMemoryStore**: Flat 50-60% across all phases (no learning)
- **Learning system**: 60% → 75% → 85% → 80% (clear improvement curve)

---

## LOCOMO-HARD Test Design

### Dataset Construction
```python
for each conversation in LOCOMO:
    real_data = conversation['conversation']

    # 1. Inject semantic noise (4× volume)
    noise_data = generate_semantic_noise(real_data, multiplier=4)

    # 2. Inject temporal contradictions (20% of facts get updated/contradicted)
    contradictory_data = inject_contradictions(real_data, rate=0.2)

    # 3. Add cross-conversation interference (3 similar conversations)
    interference_conversations = generate_similar_conversations(real_data, count=3)

    # Store ALL data (real + noise + contradictions + interference)
    combined_dataset = real_data + noise_data + contradictory_data + interference_conversations
```

### Testing Protocol
```python
# Phase 1: Cold start (no learning)
results_phase1 = test_questions(Q1_to_Q20, record_outcomes=False)

# Phase 2: Early learning
results_phase2 = test_questions(Q21_to_Q50, record_outcomes=True)

# Phase 3: Mature learning
results_phase3 = test_questions(Q51_to_Q100, record_outcomes=True)

# Phase 4: Generalization test
results_phase4 = test_questions(Q101_to_Q150, record_outcomes=True, novel=True)

# Compare learning curves
compare_systems([SimpleMemoryStore, UnifiedMemorySystem])
```

---

## Expected Results

### SimpleMemoryStore (No Learning)
```
Phase 1: 55% (semantic confusion dominates)
Phase 2: 57% (slight variance, no improvement)
Phase 3: 56% (no learning, flat performance)
Phase 4: 54% (still confused)
Average: 55.5%
```

### UnifiedMemorySystem (With Learning)
```
Phase 1: 60% (cold start, similar to Simple)
Phase 2: 74% (+14% from outcome learning)
Phase 3: 86% (+26% from strong patterns)
Phase 4: 82% (+22% patterns generalize)
Average: 75.5% (+20 points vs Simple)
```

### Mem0 (Baseline - Expected)
```
Average: ~65-70% (graph-based helps with interference, but no outcome learning)
```

---

## What LOCOMO-HARD Tests

✅ **Outcome-based learning**: Does the system improve from feedback?
✅ **Noise filtering**: Can it distinguish real vs fake similar data?
✅ **Source prioritization**: Does it learn which collections/sources work?
✅ **Temporal reasoning**: Can it handle outdated information?
✅ **Cross-context routing**: Does KG routing prevent confusion?
✅ **Learning curve**: Clear improvement over time vs flat performance

---

## Implementation Plan

### Step 1: Noise Generator
```python
def generate_semantic_noise(real_conversation: Dict, multiplier: int = 4) -> List[Dict]:
    """Generate semantically similar but factually wrong conversations"""
    # Use LLM to generate variations:
    # - Swap names (John ↔ Mike, Maria ↔ Sarah)
    # - Swap entities (yoga ↔ gym, Python ↔ JavaScript)
    # - Keep structure and topics (maintain semantic similarity)
```

### Step 2: Contradiction Injector
```python
def inject_contradictions(conversation: Dict, rate: float = 0.2) -> List[Dict]:
    """Add old/outdated versions of facts that changed"""
    # Find facts that evolve (jobs, locations, activities)
    # Create earlier versions with wrong values
```

### Step 3: Enhanced Test Runner
```python
async def run_locomo_hard(memory_system):
    # Ingest real + noise + contradictions
    # Test in phases with outcome recording
    # Track learning curve
    # Compare vs baseline (no learning)
```

---

## Why This Matters

**Current LOCOMO**: Tests retrieval from clean dataset → Learning provides no advantage

**LOCOMO-HARD**: Tests retrieval from noisy, contradictory, real-world dataset → Learning systems dominate

**This is what you built for**: Roampal's 5-tier + KG + outcome learning shines when there's:
- Noise to filter
- Sources to prioritize
- Patterns to learn
- Outcomes to adapt from

**Benchmark should match the problem your system solves.**
