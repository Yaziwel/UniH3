First, download the [MedIR-2D-500K](https://huggingface.co/datasets/upyzwup/MedIR-2D-500K) dataset and extract all `.tar` shards into the same directory. The resulting directory structure is as follows:

```
MedIR-2D-500K/
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
├── OCT/
│   ├── train/
│   │   ├── HQ/
│   │   └── LQ/
│   └── test/
│       ├── HQ/
│       └── LQ/
├── Pathology/
│   ├── train/
│   │   ├── HQ/
│   │   └── LQ/
│   └── test/
│       ├── HQ/
│       └── LQ/
├── PET/
│   ├── train/
│   │   ├── HQ/
│   │   └── LQ/
│   └── test/
│       ├── HQ/
│       └── LQ/
├── Ultrasound/
│   ├── train/
│   │   ├── HQ/
│   │   └── LQ/
│   └── test/
│       ├── HQ/
│       └── LQ/
└── X-ray/
    ├── train/
    │   ├── HQ/
    │   └── LQ/
    └── test/
        ├── HQ/
        └── LQ/
```

