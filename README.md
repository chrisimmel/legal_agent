# Legal Research Agent

An intelligent multi-agent system for legal document research and analysis. The system uses advanced AI agents to plan, execute, and reflect on research tasks to provide comprehensive answers to legal queries.

## Features

- **Multi-Agent Architecture**: Planner, Executor, and Reflector agents work together
- **Intelligent Planning**: Automatically creates step-by-step research plans
- **Vector Database Search**: Semantic search across legal documents
- **Document Processing**: PDF to markdown conversion with OCR support
- **Iterative Refinement**: Agents reflect on and improve their work
- **Comprehensive Analysis**: Synthesizes information across multiple documents

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Set up Environment Variables
Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_key
```

### 3. Run a Legal Research Query
```bash
python orchestrator.py "Cite an example of a case involving breach of contract, and tell me who prevailed in the case."
```

## How It Works

The system employs a basic multi-agent approach:

1. **Planning Phase**: The `PlannerAgent` analyzes your query and creates a step-by-step research plan
2. **Execution Phase**: The `StepExecutorAgent` performs document searches and analysis for each step
3. **Reflection Phase**: The `StepReflectorAgent` evaluates the quality of each step and decides if retry is needed
4. **Orchestration**: The `AgentOrchestrator` coordinates all agents until the research is complete

## Components

### Core Agents
- **`agents/planner.py`** - Plans multi-step research strategies
- **`agents/step_executor.py`** - Executes research steps with vector search
- **`agents/step_reflector.py`** - Evaluates and refines agent outputs
- **`orchestrator.py`** - Coordinates the entire research process

### Document Processing
- **`pdf_to_markdown_ocr.py`** - Enhanced PDF converter with OCR support
- **`document_manager.py`** - Manages document loading and processing
- **`vector_store_manager.py`** - Handles vector database operations

## Usage Examples

### Basic Research Query
```bash
python orchestrator.py "Summarize the key findings from all legal cases"
```

### Complex Multi-Step Analysis
```bash
python orchestrator.py "What types of legal proceedings are represented and what were their outcomes?"
```

### Document-Specific Research
```bash
python orchestrator.py "What are the main arguments in cases involving contract disputes?"
```

## Example Queries

- **Case Analysis**: "What are the main legal issues discussed in the case documents?"
- **Outcome Summary**: "Summarize the key findings and outcomes from all cases"  
- **Comparative Analysis**: "Compare the legal arguments across different case types"
- **Procedural Questions**: "What types of legal proceedings are represented in the database?"
- **Topic Research**: "Find all references to intellectual property disputes"

## Setup and Configuration

### Prerequisites
- Python 3.11+
- uv package manager
- OpenAI API key (for embeddings and LLM access)

### Document Preparation
1. Place PDF documents in `data/` directory
2. Run PDF processing to convert to markdown:
```bash
python pdf_to_markdown_ocr.py
```
3. Documents will be processed into `data/documents/` as markdown files


## Architecture

### Multi-Agent System
The research agent uses three specialized AI agents:

1. **PlannerAgent** (`agents/planner.py`)
   - Analyzes user queries and creates research plans
   - Breaks complex questions into manageable steps
   - Considers document types and research scope

2. **StepExecutorAgent** (`agents/step_executor.py`)
   - Executes individual research steps
   - Performs semantic search across document vector store
   - Synthesizes findings into response documents

3. **StepReflectorAgent** (`agents/step_reflector.py`)
   - Evaluates the quality and completeness of each step
   - Determines if steps need to be retried or refined
   - Ensures high-quality research outputs

### Project Structure

```
├── orchestrator.py          # Main entry point and agent coordination
├── models.py                # Data models for agent state and responses
├── llm.py                   # LLM interface and configuration
├── agents/
│   ├── base.py              # Base agent class
│   ├── planner.py           # Research planning agent
│   ├── step_executor.py     # Step execution agent
│   └── step_reflector.py    # Quality assurance agent
├── storage/
│   ├── document_manager.py      # Document loading and management
│   ├── pdf_ingest.py            # Workflow to ingest PDF documents to vector store.
│   ├── pdf_to_markdown_ocr.py   # PDF processing with OCR (unused)
│   ├── vector_database.py       # Vector database operations
│   ├── vector_store_manager.py  # Vector database operations
├── data/
│   ├── documents/           # Processed markdown documents
└── chroma_db/              # Persistent vector database
```

## Example Session

```bash
python orchestrator.py "Cite an example of a case involving breach of contract, and tell me who prevailed in the case."
```

Output:
```
📚 Loading documents...
Loading 10 documents...

...

📋 Generating execution plan...
Making LLM call with model openai/gpt-4.1
✓ Created plan with 3 steps:
  1. Conduct a vector search to identify legal case documents specifically involving breach of contract.
  2. Select an example case from the search results, summarize the facts, the legal issue, and determine which party prevailed.
  3. Present a concise, clearly formatted response citing the chosen breach of contract case and stating who prevailed.

⚡ Executing plan...

🔄 Executing step 1/3: Conduct a vector search to identify legal case documents specifically involving breach of contract.
   (Attempt 1/3)

...

🔄 Reflecting on step 3 (Present a concise, clearly formatted response citing the chosen breach of contract case and stating who prevailed.). Attempt 1.
Making LLM call with model openai/gpt-4.1
Step 2 failed
Reflector notes: The CURRENT_STEP required presenting a concise, clearly formatted response citing an example breach of contract case and stating who prevailed. The response cites a case (Li v. Singapore Telecommunications Ltd.) and clearly states the facts, legal issue, and prevailing party. However, the result indicates this case was not sourced from the current search or provided legal documents—rather, it was supplied from prior knowledge or internal memory because no relevant breach of contract case was found in the available set.

The step's objective was to use information from the actual search/documents, not from prior knowledge. Therefore, the step is unsuccessful. To improve, the executor should explicitly state that no breach of contract case was found in the retrieved materials and that it was not possible to fulfill the user query based on the available sources, rather than providing an unsourced or 'stand-in' example. If appropriate, the response should transparently mention the limitation and avoid fabricating or inserting cases not tied to retrieved evidence.
   ⚠️  Step needs retry: The CURRENT_STEP required presenting a concise, clearly formatted response citing an example breach of contract case and stating who prevailed. The response cites a case (Li v. Singapore Telecommunications Ltd.) and clearly states the facts, legal issue, and prevailing party. However, the result indicates this case was not sourced from the current search or provided legal documents—rather, it was supplied from prior knowledge or internal memory because no relevant breach of contract case was found in the available set.

The step's objective was to use information from the actual search/documents, not from prior knowledge. Therefore, the step is unsuccessful. To improve, the executor should explicitly state that no breach of contract case was found in the retrieved materials and that it was not possible to fulfill the user query based on the available sources, rather than providing an unsourced or 'stand-in' example. If appropriate, the response should transparently mention the limitation and avoid fabricating or inserting cases not tied to retrieved evidence.

...

Step 2 completed successfully
Reflector notes: The executor has successfully completed the current step. The response is concise, clearly formatted, and cites a specific breach of contract case, including relevant facts, the legal issue, the outcome, and who prevailed (Ms. Li). The arbitrator's decision and key reasoning are summarized, and the case citation is provided. The response fulfills the user query and the objectives of the step, with no significant omissions or errors.
   ✓ Step completed successfully

🎉 Plan execution completed!

============================================================
FINAL RESPONSE:
============================================================
Example Case: Li v. Singapore Telecommunications Ltd. (SIAC-2021-045)

Facts: Ms. Li was employed by Singapore Telecommunications Ltd. (SingTel) on an expatriate contract guaranteeing a 60-day notice period or equivalent compensation if terminated. SingTel terminated Li, alleging breach of a confidentiality clause (Art. 15) by claiming she worked remotely without approval. Li asserted she had fulfilled her reporting duties and that the notice requirement was mandatory.

Legal Issue: Whether SingTel was justified in summarily dismissing Li for the alleged breach, thus avoiding the contractual obligation to give notice or pay compensation.

Decision and Prevailing Party: The arbitrator, applying Singapore Contract Law, found no substantive evidence of unauthorized disclosure by Li and further noted that SingTel failed to follow its own progressive discipline procedures. The award ordered SingTel to pay Ms. Li SGD 180,000 for notice and lost benefits, plus costs. Therefore, Ms. Li prevailed in the case.

Citation: SIAC Arbitration Case No. SIAC-2021-045, Li v. Singapore Telecommunications Ltd., Award dated May 27, 2021.
============================================================


```

Notice that the agents reflect on the result of each step and are prepared to retry as necessary (see "⚠️  Step needs retry" in the log).



## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure ANTHROPIC_API_KEY is set
2. **No Documents Found**: Check that documents exist in `data/documents/`
3. **Vector Store Empty**: Run document processing pipeline first
4. **Step Execution Failures**: Check document quality and query complexity

### Performance Optimization
- Use OpenAI embeddings for better search quality
- Adjust chunk sizes based on document types
- Monitor agent execution logs for bottlenecks
- Consider document preprocessing for complex layouts

### Debug Mode
Run with verbose output to see agent decision-making:
```bash
python orchestrator.py "your query" --verbose
```

## License

This code is provided as-is for educational and research purposes.
