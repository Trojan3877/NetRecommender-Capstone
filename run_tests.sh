#!/bin/bash
echo "Running tests with coverage..."
coverage run -m unittest discover -s tests
coverage report -m
coverage html  # Generates HTML report in ./htmlcov
