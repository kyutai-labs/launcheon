#!/bin/bash
set -e
echo
echo
echo -e "\e[31m[1/3] Checking linting and formatting\e[0m"
uv run --group dev --frozen ruff format launcheon;
uv run --group dev --frozen ruff check --fix --extend-select I launcheon; 

echo 
echo
echo -e "\e[31m[2/3] Checking typing\e[0m"
uv run --group dev --frozen  -m pyright launcheon


echo 
echo
echo -e "\e[31m[3/3] Unit tests\e[0m"
uv run --group dev --frozen -m pytest tests