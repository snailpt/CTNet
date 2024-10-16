# CTNet for motor imagery EEG classification
### CTNet: A Convolutional Transformer Network for EEG-Based Motor Imagery Classification [[Paper](https://www.nature.com/articles/s41598-024-71118-7)]
core idea: CNN (an improved version of EEGNet) + Transformer encoder 

Our research builds upon and improves the [EEG Conformer](https://github.com/eeyhsong/EEG-Conformer) and [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet), and we sincerely thank the creators of these open-source project.

### News
🎉🎉🎉 We've joined in [braindecode(https://github.com/braindecode/braindecode/)] toolbox. Use [here(https://github.com/braindecode/braindecode/blob/master/braindecode/models/ctnet.py)] for detailed info.
Thanks to [Bru(https://github.com/bruAristimunha)] and colleagues for helping with the modifications.

### Abstract:
Brain-computer interface (BCI) technology bridges the direct communication between the brain and machines, unlocking new possibilities for human interaction and rehabilitation. EEG-based motor imagery (MI) plays a pivotal role in BCI, enabling the translation of thought into actionable commands for interactive and assistive technologies. However, the constrained decoding performance of brain signals poses a limitation to the broader application and development of BCI systems. In this study, we introduce a convolutional Transformer network (CTNet) designed for EEG-based MI classification. Firstly, CTNet employs a convolutional module analogous to EEGNet, dedicated to extracting local and spatial features from EEG time series. Subsequently, it incorporates a Transformer encoder module, leveraging a multi-head attention mechanism to discern the global dependencies of EEG's high-level features. Finally, a straightforward classifier module comprising fully connected layers is followed to categorize EEG signals. In subject-specific evaluations, CTNet achieved remarkable decoding accuracies of 82.52% and 88.49% on the BCI IV-2a and IV-2b datasets, respectively. Furthermore, in the challenging cross-subject assessments, CTNet achieved recognition accuracies of 58.64% on the BCI IV-2a dataset and 76.27% on the BCI IV-2b dataset. In both subject-specific and cross-subject evaluations, CTNet holds a leading position when compared to some of the state-of-the-art methods. This underscores the exceptional efficacy of our approach and its potential to set a new benchmark in EEG decoding.


### Overall Framework:
![architecture of CTNet](https://raw.githubusercontent.com/snailpt/CTNet/main/architecture.png)

### Requirements:
Python 3.10

Pytorch 1.13.1


### Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊

Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7
