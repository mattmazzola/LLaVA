git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Install package locally
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install additional packages for training
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

echo "postCreateCommand.sh COMPLETE!"
