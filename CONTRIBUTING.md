# Contributing to Multi-Hop RAG

Thank you for your interest in contributing to the Multi-Hop RAG project! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

### Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/multi-hop-rag.git
cd multi-hop-rag
```

3. Set up development environment:
```bash
# Using UV (recommended)
./setup_uv.sh

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

4. Create a branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Run these before committing:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Code Standards

1. **Type Hints**: Use type hints for all function parameters and return values
```python
def process_document(text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
    ...
```

2. **Docstrings**: Use Google-style docstrings
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
```

3. **Logging**: Use the logging module, not print statements
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing document")
```

4. **Error Handling**: Use specific exceptions and meaningful error messages
```python
if not file_path.exists():
    raise FileNotFoundError(f"Document not found: {file_path}")
```

### Testing

1. **Write Tests**: All new features should include tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_chunking.py

# Run with coverage
pytest tests/ --cov=src/multi_hop_rag --cov-report=html
```

2. **Test Structure**:
   - Unit tests in `tests/test_*.py`
   - Integration tests marked with `@pytest.mark.integration`
   - Mock external API calls in unit tests

3. **Test Guidelines**:
   - One test file per module
   - Clear test names: `test_function_name_expected_behavior`
   - Use fixtures for common setup
   - Test both success and failure cases

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Longer description if needed

- Detail 1
- Detail 2
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Maintenance tasks

Examples:
```
feat(chunking): add support for markdown documents
fix(embedder): handle empty text gracefully
docs(readme): update installation instructions
```

## Pull Request Process

1. **Before Submitting**:
   - Run tests: `pytest tests/`
   - Run linters: `black src/ && ruff check src/`
   - Update documentation if needed
   - Add tests for new features

2. **PR Description**:
   - Describe what the PR does
   - Reference related issues
   - Include screenshots for UI changes
   - Note any breaking changes

3. **PR Checklist**:
   - [ ] Tests pass locally
   - [ ] Code is formatted (black)
   - [ ] Code is linted (ruff)
   - [ ] Documentation updated
   - [ ] Commit messages follow convention
   - [ ] Branch is up to date with main

## Areas for Contribution

### High Priority

- **Performance Optimization**: Improve embedding/retrieval speed
- **Additional Document Formats**: Support for HTML, Markdown, etc.
- **Better Error Handling**: More robust error recovery
- **Monitoring**: Add metrics and observability

### Good First Issues

- **Documentation**: Improve examples and guides
- **Testing**: Increase test coverage
- **Bug Fixes**: Address open issues
- **Type Hints**: Add missing type annotations

### Feature Ideas

- **Streaming Responses**: Stream LLM responses
- **Caching**: Cache embeddings and LLM responses
- **Multi-language Support**: Support non-English documents
- **Custom Retrievers**: Pluggable retrieval strategies
- **Web Interface**: Add a web UI

## Development Workflow

### Adding a New Feature

1. Create an issue describing the feature
2. Get feedback from maintainers
3. Create a branch: `feature/feature-name`
4. Implement the feature with tests
5. Update documentation
6. Submit a PR

### Fixing a Bug

1. Create an issue if one doesn't exist
2. Create a branch: `fix/bug-description`
3. Write a failing test that reproduces the bug
4. Fix the bug
5. Ensure test passes
6. Submit a PR

### Example Contribution Flow

```bash
# 1. Create branch
git checkout -b feat/add-html-support

# 2. Make changes
# ... edit files ...

# 3. Test changes
pytest tests/
black src/ tests/
ruff check src/ tests/

# 4. Commit changes
git add .
git commit -m "feat(chunking): add HTML document support"

# 5. Push and create PR
git push origin feat/add-html-support
# Then create PR on GitHub
```

## Project Structure

```
multi-hop-rag/
├── src/multi_hop_rag/      # Main package
│   ├── chunking/           # Document chunking
│   ├── embedding/          # Embedding generation
│   ├── indexing/           # Vector storage
│   ├── retrieval/          # Multi-hop retrieval
│   ├── graph/              # LangGraph implementation
│   ├── config.py           # Configuration
│   └── pipeline.py         # Main pipeline
├── tests/                  # Tests
├── examples/               # Example scripts
├── docs/                   # Documentation (future)
└── pyproject.toml          # Project configuration
```

## Architecture Decisions

When making significant changes, consider:

1. **Backward Compatibility**: Avoid breaking existing APIs
2. **Performance**: Profile before optimizing
3. **Dependencies**: Minimize new dependencies
4. **Testability**: Design for easy testing
5. **Documentation**: Update docs with code changes

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with reproduction steps
- **Features**: Open an issue for discussion first

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page

Thank you for contributing to Multi-Hop RAG! 🎉
