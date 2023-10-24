git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Set AZCOPY concurrency to auto
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.zshrc
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.bashrc

# Activate conda by default
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.bashrc

# Use llava environment by default
echo "conda activate llava" >> ~/.zshrc
echo "conda activate llava" >> ~/.bashrc

# Add dotnet to PATH
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.zshrc

# Create and activate llava environment
source /home/vscode/miniconda3/bin/activate
conda create -y -q -n llava python=3.10
conda activate llava

# Install Nvidia Cuda Compiler
conda install -y -c nvidia cuda-compiler

pip install pre-commit==3.0.2

# Install package locally
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install additional packages for training
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

echo "postCreateCommand.sh COMPLETE!"
