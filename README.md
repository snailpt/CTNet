# CTNet for motor imagery EEG classification
### CTNet: A Convolutional Transformer Network for EEG-Based Motor Imagery Classification [[Paper](https://www.nature.com/articles/s41598-024-71118-7)]
core idea: CNN (an improved version of EEGNet) + Transformer encoder 

Our research builds upon and improves the [EEG Conformer](https://github.com/eeyhsong/EEG-Conformer) and [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet), and we sincerely thank the creators of these open-source project.

### News
🎉🎉🎉 We've joined in [braindecode toolbox](https://github.com/braindecode/braindecode/). Use [here](https://github.com/braindecode/braindecode/blob/master/braindecode/models/ctnet.py) for detailed info.
Thanks to [Bru](https://github.com/bruAristimunha) and colleagues for helping with the modifications.

### Abstract:
Brain-computer interface (BCI) technology bridges the direct communication between the brain and machines, unlocking new possibilities for human interaction and rehabilitation. EEG-based motor imagery (MI) plays a pivotal role in BCI, enabling the translation of thought into actionable commands for interactive and assistive technologies. However, the constrained decoding performance of brain signals poses a limitation to the broader application and development of BCI systems. In this study, we introduce a convolutional Transformer network (CTNet) designed for EEG-based MI classification. Firstly, CTNet employs a convolutional module analogous to EEGNet, dedicated to extracting local and spatial features from EEG time series. Subsequently, it incorporates a Transformer encoder module, leveraging a multi-head attention mechanism to discern the global dependencies of EEG's high-level features. Finally, a straightforward classifier module comprising fully connected layers is followed to categorize EEG signals. In subject-specific evaluations, CTNet achieved remarkable decoding accuracies of 82.52% and 88.49% on the BCI IV-2a and IV-2b datasets, respectively. Furthermore, in the challenging cross-subject assessments, CTNet achieved recognition accuracies of 58.64% on the BCI IV-2a dataset and 76.27% on the BCI IV-2b dataset. In both subject-specific and cross-subject evaluations, CTNet holds a leading position when compared to some of the state-of-the-art methods. This underscores the exceptional efficacy of our approach and its potential to set a new benchmark in EEG decoding.


### Overall Framework:
![architecture of CTNet](https://raw.githubusercontent.com/snailpt/CTNet/main/architecture.png)

### Requirements:
Python 3.10

Pytorch 1.13.1
Cross-subject classification accuracy (in percentage %) and kappa of state-of-the-art algorithms on the BCI IV-2a dataset.
| Method               | A01   | A02   | A03   | A04   | A05   | A06   | A07   | A08   | A09   | Average±Std.    | Kappa  |
|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-----------------|--------|
| ShallowConvNet+[27]  | 66.84 | 46.53 | 67.53 | 52.26 | 34.38 | 39.76 | 65.45 | 71.18 | 66.84 | 56.75±13.77     | 0.4234 |
| DeepConvNet+[27]     | 68.58 | 47.40 | 78.99 | 52.26 | 50.87 | 41.84 | 69.44 | 71.70 | 60.24 | 60.15±12.71     | 0.4686 |
| EEGNet+[28]          | 69.79 | 42.01 | 79.51 | 50.87 | 35.76 | 37.15 | 65.80 | 67.36 | 63.37 | 56.85±15.82     | 0.4246 |
| Conformer+[53]       | 68.75 | 37.33 | 69.62 | 43.58 | 29.51 | 35.24 | 58.33 | 74.48 | 63.89 | 53.41±17.08     | 0.3789 |
| CTNet (Proposed)     | 69.27 | 43.92 | 79.34 | 55.38 | 43.92 | 36.11 | 65.10 | 70.66 | 64.06 | 58.64±14.61     | 0.4486 |


Cross-subject classification accuracy (in percentage %) and Kappa of state-of-the-art algorithms on the BCI IV-2b dataset.
| Method               | B01   | B02   | B03   | B04   | B05   | B06   | B07   | B08   | B09   | Average±Std.    | Kappa  |
|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-----------------|--------|
| ShallowConvNet+[27]  | 74.03 | 63.53 | 59.72 | 82.84 | 82.43 | 80.97 | 74.86 | 72.37 | 77.78 | 74.28±8.13      | 0.4856 |
| DeepConvNet+[27]     | 74.03 | 65.15 | 63.47 | 80.81 | 82.70 | 74.86 | 81.39 | 76.32 | 77.92 | 75.18±6.84      | 0.5037 |
| EEGNet+[28]          | 74.44 | 69.26 | 62.36 | 80.41 | 83.24 | 75.56 | 79.86 | 73.55 | 77.50 | 75.13±6.35      | 0.5026 |
| Conformer+[53]       | 71.39 | 62.35 | 65.28 | 82.97 | 80.41 | 69.31 | 75.00 | 76.32 | 78.61 | 73.52±6.96      | 0.4703 |
| CTNet (Proposed)     | 76.25 | 71.03 | 66.39 | 81.76 | 83.11 | 77.22 | 79.17 | 73.56 | 77.92 | 76.27±5.26      | 0.5252 |


### Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊

Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7
