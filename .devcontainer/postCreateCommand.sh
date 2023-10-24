git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Set AZCOPY concurrency to auto
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.zshrc
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.bashrc

source /home/vscode/miniconda3/bin/activate

# Create and activate llava environment
conda create -y -q -n llava python=3.10
conda activate llava

echo ". /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.bashrc
echo "conda activate llava" >> ~/.zshrc
echo "conda activate llava" >> ~/.bashrc

pip install pre-commit==3.0.2

echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.zshrc

# Setup CUDA_HOME for deepspeed
echo "export CUDA_HOME=/usr/lib/x86_64-linux-gnu/" >> ~/.bashrc
echo "export CUDA_HOME=/usr/lib/x86_64-linux-gnu/" >> ~/.zshrc

# Install package locally
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install additional packages for training
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

echo "postCreateCommand.sh COMPLETE!"
