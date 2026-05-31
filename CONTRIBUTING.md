# Contributing

Thanks for contributing to this project.

## Development Setup

1. Create a virtual environment.
2. Install dependencies from requirements.txt.
3. Run a smoke training command before opening a pull request.

## Recommended Local Validation

```bash
python -m compileall src
python src/main.py --epochs 1 --n-samples 256 --batch-size 64 --scheduler none --no-amp --artifacts-dir artifacts/local-smoke
```

## Pull Requests

Use the pull request template and keep each pull request focused on a single change set.

## Reporting Issues

Use the issue templates to provide reproducible details.
