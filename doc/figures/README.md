# Figures Directory

This directory should contain the following figures for the Euro-Par paper:

## Required Figures

### Figure 1: GPU Load Imbalance
**File**: `fig_imbalance.pdf` or `fig_imbalance.png`
**Caption**: Distribution of computational load across 8 GPUs under uniform TP partitioning.
**Content**: Bar chart showing FLOPs distribution across GPUs, with one GPU showing 45% more load than others.

### Figure 2: System Architecture
**File**: `fig_overview.pdf` or `fig_overview.png`
**Caption**: System architecture showing the three-level solver.
**Content**: Diagram showing:
- Level 1: Offline neuron partitioning
- Level 2: GPU-internal TC/CC mapping
- Level 3: Online pipeline scheduling
- Data flow between components

### Figure 3: Scalability Analysis
**File**: `fig_scale.pdf` or `fig_scale.png`
**Caption**: Scalability analysis with increasing number of GPUs.
**Content**: Line chart showing speedup vs number of GPUs (2, 4, 8, 16) for different methods.

## Figure Format Requirements

- **Format**: PDF (preferred) or high-resolution PNG (300 dpi minimum)
- **Size**: Fit within column width (3.3 inches for single column, 7 inches for double column)
- **Font**: Readable at print size (minimum 8pt)
- **Color**: Use color-blind friendly palette
- **Labels**: Clear axis labels, legends, and annotations

## Figure Creation Guidelines

### For Bar Charts (Figure 1)
```python
import matplotlib.pyplot as plt
import numpy as np

# Example code structure
gpus = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', 'GPU 4', 'GPU 5', 'GPU 6', 'GPU 7']
flops = [145, 132, 118, 105, 98, 95, 100, 100]  # Placeholder data

plt.figure(figsize=(7, 3))
plt.bar(gpus, flops)
plt.ylabel('Relative FLOPs (%)')
plt.xlabel('GPU')
plt.title('Computational Load Distribution')
plt.savefig('fig_imbalance.pdf', bbox_inches='tight', dpi=300)
```

### For System Diagram (Figure 2)
Use tools like:
- Draw.io (free)
- PowerPoint/Keynote
- LaTeX TikZ
- Adobe Illustrator

Include clear component boxes with arrows showing data flow.

### For Line Charts (Figure 3)
```python
import matplotlib.pyplot as plt
import numpy as np

# Example code structure
gpus = [2, 4, 8, 16]
megatron = [2.0, 3.8, 7.0, 12.5]
dejavu = [2.2, 4.2, 7.8, 13.8]
ours = [4.0, 7.5, 14.0, 26.0]  # Placeholder data

plt.figure(figsize=(7, 4))
plt.plot(gpus, megatron, 'o-', label='Megatron-LM')
plt.plot(gpus, dejavu, 's-', label='DejaVu')
plt.plot(gpus, ours, '^-', label='Ours')
plt.xlabel('Number of GPUs')
plt.ylabel('Speedup')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fig_scale.pdf', bbox_inches='tight', dpi=300)
```

## Color Palette

Use color-blind friendly palette:
- Blue: #0077BB
- Orange: #EE7733
- Green: #009988
- Red: #CC3311
- Purple: #AA3377

## Notes

- All figures should be referenced in the text before they appear
- Use vector graphics (PDF) when possible for better print quality
- Ensure text in figures is readable when printed in black and white
- Include error bars for experimental data
