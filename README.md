# FractalLearner Proof of Concept

## Overview
FractalLearner is a comprehensive, experimental learning system designed to process and analyze information through recursive, fractal-inspired methods. It supports multiple learning modes, multimedia input (text, images, audio), persistent knowledge storage, and semantic search. This proof of concept demonstrates core functionalities for adaptive, multimodal learning.

## Features
- **Recursive Learning**: Processes information in layered, recursive depths with configurable max depth.
- **Multiple Learning Modes**: Supports comparative, critical, creative, and standard analytical learning.
- **Multimedia Learning**: Handles text, image (via CLIP), and audio (via speech recognition) inputs.
- **Knowledge Persistence**: Stores knowledge base and vector database for persistent learning.
- **Semantic Search**: Uses FAISS vector database for similarity-based knowledge retrieval.
- **Interactive Sessions**: Facilitates guided learning with adaptive next-step suggestions.
- **Integration**: Supports integration with external AI systems for knowledge sharing.

## Project Structure
```
fractal_learner/
├── fractal_learner.py
└── my_knowledge_base/
    ├── knowledge_base.json
    ├── vector_db.index
    └── learner_state_*.pkl
```

## Prerequisites
- **Python**: Version 3.8+ recommended.
- **Dependencies**: Install required packages:
  ```bash
  pip install torch transformers faiss-cpu numpy pillow soundfile speechrecognition
  ```
- **Directory Setup**: Create storage directory:
  ```bash
  mkdir -p my_knowledge_base
  ```

## Installation
1. Clone or create the project directory:
   ```bash
   mkdir fractal_learner
   cd fractal_learner
   ```
2. Save the provided `fractal_learner.py` in the project directory.
3. Install dependencies:
   ```bash
   pip install torch transformers faiss-cpu numpy pillow soundfile speechrecognition
   ```
4. Create the knowledge base directory:
   ```bash
   mkdir -p my_knowledge_base
   ```

## Usage
Run the FractalLearner script to start learning or interacting with the system.

### Example Commands
1. **Interactive Learning**:
   ```python
   from fractal_learner import FractalLearner

   learner = FractalLearner(max_depth=6, storage_path="./my_knowledge_base")
   session_id = learner.start_interactive_session("science_tutorial")
   response = learner.interactive_learn(
       session_id,
       "Explain quantum entanglement in simple terms",
       learning_mode="comparative"
   )
   print("Learning insights:")
   for insight in response['insights']:
       print(f"Depth {insight['depth']}: {insight['insight']}")
   print("\nSuggested next steps:")
   for step in response['next_steps']:
       print(f"- {step}")
   ```

2. **Multimedia Learning**:
   ```python
   learner.learn_from_image("physics_diagram.jpg", "Quantum mechanics illustration")
   learner.learn_from_audio("lecture.wav")
   ```

3. **Semantic Search**:
   ```python
   results = learner.semantic_search("quantum physics", k=3)
   print(f"Semantic search results: {len(results)} found")
   for result in results:
       print(f"- {result['text']} (Similarity: {result['similarity']:.2f})")
   ```

4. **Save State**:
   ```python
   learner.save_state()
   ```

## Testing Key Features
- **Recursive Learning**: Observe layered insights generated for input text in different modes.
- **Multimedia Processing**: Test image and audio inputs to see extracted knowledge.
- **Semantic Search**: Query the knowledge base to retrieve relevant insights.
- **Interactive Sessions**: Start a session and provide inputs to see adaptive learning paths.
- **Persistence**: Save and load the learner state to verify knowledge retention.

## Customization
- Modify `max_depth` in `FractalLearner` initialization to adjust learning recursion.
- Add new learning modes by extending `learning_modes` in `fractal_learner.py`.
- Adjust embedding generation in `_get_embedding` for more sophisticated vectorization (e.g., use sentence transformers).
- Customize storage paths or file formats in `my_knowledge_base`.

## Dependencies
- `torch`
- `transformers`
- `faiss-cpu`
- `numpy`
- `pillow`
- `soundfile`
- `speechrecognition`
