First, download the [MedIR-3D-3K](https://huggingface.co/datasets/upyzwup/MedIR-3D-3K) dataset. The resulting directory structure is as follows:

```
MedIR-3D-3K/
├── CT/
│   ├── train/
│   │   ├── HQ/
│   │   └── LQ/
│   └── test/
│       ├── HQ/
│       └── LQ/
├── MRI/
│   ├── train/
│   │   ├── HQ/
│   │   └── LQ/
│   └── test/
│       ├── HQ/
│       └── LQ/
└── PET/
    ├── train/
    │   ├── HQ/
    │   └── LQ/
    └── test/
        ├── HQ/
        └── LQ/
```

Since 3D volumes are large, we train on smaller patches rather than entire volumes. To speed up data loading, pre-extract the patches by running `generate_3D_patches.py`:

```
python generate_3D_patches.py
```

