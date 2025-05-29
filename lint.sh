#!/bin/bash

# Run black, isort, mypy, flake8, and autopep8 on the mbr directory

# Check if --fix flag is provided
FIX=false
if [ "$1" == "--fix" ]; then
    FIX=true
fi

if [ "$FIX" = true ]; then
    echo "Running black (with autofix)..."
    black mbr/
    
    echo "Running isort (with autofix)..."
    isort mbr/
    
    echo "Running autopep8 (to fix flake8 issues)..."
    autopep8 --in-place --recursive --aggressive --aggressive mbr/
else
    echo "Running black (check only)..."
    black --check mbr/
    
    echo "Running isort (check only)..."
    isort --check mbr/
fi

echo "Running mypy..."
mypy mbr/

echo "Running flake8..."
flake8 mbr/

echo "Done!"
echo ""
if [ "$FIX" = false ]; then
    echo "To automatically fix issues, run: ./lint.sh --fix"
fi
