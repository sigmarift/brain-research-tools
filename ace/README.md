# Put the ACE library files here

This repo does **not** include WorldQuant's ACE API files. Every tool in `tools/` looks for them in this folder.

1. Download the ACE API package from the BRAIN platform.
2. Copy these two files into this folder:
   - `ace_lib.py`
   - `helpful_functions.py`
3. Restart your notebook kernel.

```
ace/
├── README.md              (this file)
├── ace_lib.py             <- you add
└── helpful_functions.py   <- you add
```

They are listed in `.gitignore`, so they can't be committed by accident.

**Other locations also work:** next to a tool's own files, or any folder you name in the `ACE_LIB_DIR` environment variable.

**Offline mode** (building and dry-running alphas without the BRAIN API) works without these files.
