# Code Documentation
Generated on: 2025-01-15T20:24:35.097Z
Total files: 9

## Project Structure

```
└── language_detection
    ├── README.md
    ├── __init__.py
    ├── config
    │   └── settings.py
    ├── core
    │   ├── detector.py
    │   └── preprocessor.py
    ├── docs
    │   └── doc-tree.md
    ├── scripts
    │   └── mas.sh
    ├── tests
    │   └── test_detector.py
    └── utils
        └── text_utils.py
```

## File: README.md
- Path: `/root/git/language_detection/README.md`
- Size: 543.00 B
- Extension: .md
- Lines of code: 16

```md
```mermaid
flowchart LR
A[Raw Text Input] --> B[Preprocessing & Normalization]
B --> C{Is Text Empty?}
C -- "Yes" --> M[Label = EMPTY]
C -- "No" --> D{Short Text?}
D -- "Yes" --> E[Optional Dictionary-Based Detection]
D -- "No" --> H[fastText Model Prediction]
E --> F{Confident Dictionary Result?}
F -- "No" --> H
F -- "Yes" --> G[Label = {ar, fr, en}]
H --> I{Confidence >= Threshold?}
I -- "No" --> J[Label = UNKNOWN]
I -- "Yes" --> K[Label = {ar, fr, en}]
M & G & J & K --> L[End]
```
```

---------------------------------------------------------------------------

## File: __init__.py
- Path: `/root/git/language_detection/__init__.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: settings.py
- Path: `/root/git/language_detection/config/settings.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: detector.py
- Path: `/root/git/language_detection/core/detector.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: preprocessor.py
- Path: `/root/git/language_detection/core/preprocessor.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: doc-tree.md
- Path: `/root/git/language_detection/docs/doc-tree.md`
- Size: 294.00 B
- Extension: .md
- Lines of code: 20

```md
# Project Tree Structure
```plaintext
.
|-- config
|   `-- settings.py
|-- core
|   |-- detector.py
|   `-- preprocessor.py
|-- data
|-- docs
|-- scripts
|   `-- mas.sh
|-- tests
|   `-- test_detector.py
|-- utils
|   `-- text_utils.py
|-- __init__.py
`-- README.md
7 directories, 8 files
```
```

---------------------------------------------------------------------------

## File: mas.sh
- Path: `/root/git/language_detection/scripts/mas.sh`
- Size: 495.00 B
- Extension: .sh
- Lines of code: 18

```sh
#!/bin/bash
echo "Generating documentation Config..."
mkdir -p docs
# Generate tree structure with specific ignores
TREE_OUTPUT=$(tree -a -I 'node_modules|.git|.next|dist|.turbo|.cache|.vercel|coverage' \
--dirsfirst \
--charset=ascii)
{
echo "# Project Tree Structure"
echo "\`\`\`plaintext"
echo "$TREE_OUTPUT"
echo "\`\`\`"
} > docs/doc-tree.md
cw doc \
--pattern "." \
--output "docs/doc.md" \
--compress false
echo "Documentation generated successfully."
```

---------------------------------------------------------------------------

## File: test_detector.py
- Path: `/root/git/language_detection/tests/test_detector.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: text_utils.py
- Path: `/root/git/language_detection/utils/text_utils.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------