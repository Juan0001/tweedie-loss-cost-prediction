# Contributing

Thank you for your interest in contributing to this project! Here's how you can help.

## Ways to Contribute

- **Bug reports**: Open an issue describing the problem, including notebook name, cell number, and error message.
- **New use cases**: Add a notebook demonstrating Tweedie loss in a new domain (e.g., energy consumption, network traffic, ad spend).
- **Framework extensions**: Add implementations for additional frameworks (e.g., TensorFlow/Keras, JAX, Ray).
- **Documentation**: Improve explanations, add diagrams, fix typos.
- **Performance**: Optimize code, add GPU benchmarks, improve scalability.

## How to Submit Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Make your changes
4. Ensure notebooks run without errors
5. Commit with a clear message (`git commit -m "Add TensorFlow Tweedie loss example"`)
6. Push to your fork (`git push origin feature/my-improvement`)
7. Open a Pull Request

## Code Style

- Python code follows PEP 8
- Notebooks should include markdown explanations between code cells
- All functions should have docstrings with parameters, returns, and examples
- Use type hints where practical

## Notebook Guidelines

- Each notebook should be self-contained (runnable top-to-bottom)
- Include `!pip install` commands (commented out) for non-standard libraries
- Use synthetic data generation when public datasets aren't available
- Include evaluation metrics and visualizations

## Questions?

Open an issue or reach out via the discussions tab.
