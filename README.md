# ReZero

## Project Organization

```
    ├── .github/               <- GitHub settings.
    │
    ├── data/                  <- Datasets.
    │
    ├── environments/          <- Environment-specific configurations.
    │
    ├── models/                <- Pretrained and serialized models.
    │
    ├── outputs/               <- Outputs.
    │
    ├── src/                   <- Python Source code.
    │
    ├── .dockerignore
    ├── .gitignore
    ├── LICENSE
    ├── uv.lock                <- Auto-generated lock file.
    ├── pyproject.toml         <- Main project configuration file.
    └── README.md              <- Top-level README for developers.
```

### ✅ Environment Variable Setup

🔧 Save Host UID and GID

Create a script to detect and persist your user and group IDs:

```bash
id -u  # e.g., 1000
id -g  # e.g., 1000
echo $USER # e.g., sakaya
```

Edit your shell config file:

```bash
vim ~/.bash_profile  # Or ~/.bashrc, depending on your shell
```

Add the following lines:

```bash
# Addendum if there are none
if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi

# Be sure to add
export HOST_UID=1000  # Replace with output from id -u
export HOST_GID=1000  # Replace with output from id -g
export USER_NAME=sakaya # Replace with output from echo $USER
```

Apply changes:

```bash
source ~/.bash_profile
```
## Activae Docker Container
Run the transcription script:

```bash
bash scripts/A6000/1st.sh # A6000 1st
bash scripts/A6000/2nd.sh # A6000 2nd
bash scripts/mic/mic.sh   # MicMachine
```

### Install gpuRIR
```bash
uv pip install git+https://github.com/DavidDiazGuerra/gpuRIR.git
```