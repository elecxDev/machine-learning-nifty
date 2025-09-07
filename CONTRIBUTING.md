# Contributing to Machine Learning Nifty

Thank you for your interest in contributing to Machine Learning Nifty! We welcome contributions from everyone.

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and inclusive in all interactions.

## 🚀 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain the behavior you expected**
- **Explain why this enhancement would be useful**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install dependencies** and set up your development environment
3. **Make your changes** following our coding standards
4. **Add tests** for any new functionality
5. **Ensure all tests pass**
6. **Update documentation** if needed
7. **Submit a pull request**

## 🛠️ Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/machine-learning-nifty.git
cd machine-learning-nifty
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## 📝 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

### Code Organization

- **Modular design**: Keep functions and classes focused and small
- **Clear naming**: Use descriptive names for variables, functions, and classes
- **Documentation**: Include docstrings for all public functions and classes
- **Type hints**: Use type hints where appropriate

### Testing

- Write unit tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names
- Include both positive and negative test cases

## 🧪 Running Tests

Run the full test suite:
```bash
python -m pytest
```

Run tests with coverage:
```bash
python -m pytest --cov=src --cov-report=html
```

Run specific tests:
```bash
python -m pytest tests/test_specific_module.py
```

## 📚 Documentation

- Update docstrings for any modified functions
- Update the README if you add new features
- Add examples for new functionality
- Use clear and concise language

### Docstring Format

We use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: If param1 is empty.
    """
    pass
```

## 🔄 Workflow

1. **Check existing issues** to see if your idea is already being discussed
2. **Create an issue** to discuss major changes before implementing
3. **Create a feature branch** from `main`
4. **Make your changes** in small, logical commits
5. **Write or update tests** for your changes
6. **Update documentation** if needed
7. **Submit a pull request** with a clear description

### Commit Messages

- Use clear and meaningful commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Reference issues when applicable (e.g., "Fixes #123")

### Branch Naming

- Use descriptive branch names
- Format: `feature/description`, `bugfix/description`, `hotfix/description`
- Examples: `feature/model-evaluation`, `bugfix/data-loading`

## 🏷️ Release Process

1. Update version number in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create a pull request for the release
4. Tag the release after merging
5. Create a GitHub release with release notes

## ❓ Questions?

If you have questions about contributing, please:

1. Check the existing issues and discussions
2. Create a new issue with the "question" label
3. Join our community discussions

## 🙏 Recognition

Contributors will be recognized in:
- The README.md file
- Release notes
- Our contributors page

Thank you for helping make Machine Learning Nifty better! 🎉
