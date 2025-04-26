from setuptools import setup, find_packages

# Base dependencies required for all installations
base_deps = [
    "torch~=2.5.0",
    "torchvision~=0.20.0",
    "numpy~=1.26.0",
    "pandas~=2.2.0",
    "Pillow~=10.2.0",
    "scipy~=1.14.0",
    "h5py~=3.12.1",
    "beaker-py~=1.34.1",
]

train_deps = [
    "wandb~=0.19.8",
    "hydra-core~=1.3.2",
    "torchmetrics~=1.6.3",
    "sentencepiece~=0.2.0",
    "transformers @ git+https://github.com/huggingface/transformers@159445d044623a4eba23ceb96dc7bd5bda51aa1a",
    "nv_embed_v2 @ git+https://github.com/abhaybd/NV-Embed-v2_fixed.git",
]

eval_deps = [
    "semantic-grasping-datagen @ git+https://github.com/abhaybd/semantic-grasping-datagen.git",
    "fastapi[standard]",
    "wandb[media]",
]

setup(
    name="semantic-grasping",
    version="0.1.0",
    description="Semantic grasping project",
    author="Abhay D",
    packages=find_packages(),
    install_requires=base_deps,
    extras_require={
        "train": train_deps,
        "eval": eval_deps,
        "all": train_deps + eval_deps,
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)