from pathlib import Path

import pandas as pd

ABSTRACTS = [
    {
        "title": "Deep learning for EEG-based brain-computer interfaces",
        "abstract": (
            "Brain-computer interfaces use neural signals to provide a communication "
            "pathway between the brain and an external device. Electroencephalography "
            "is attractive because it is non-invasive and relatively inexpensive, but "
            "EEG signals are noisy and vary substantially between users. Deep learning "
            "models can learn representations of neural activity and improve the "
            "classification of imagined movements and other mental tasks."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Machine learning methods for classifying cognitive states from EEG",
        "abstract": (
            "Electroencephalography can be used to estimate cognitive states such as "
            "attention, workload, and fatigue. This study compares conventional machine "
            "learning methods with neural network models for EEG classification. Feature "
            "quality, subject variability, and evaluation design strongly influence "
            "reported performance."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Functional magnetic resonance imaging of human memory",
        "abstract": (
            "Functional magnetic resonance imaging provides a non-invasive method for "
            "measuring changes in brain activity during memory tasks. Research using "
            "fMRI has identified distributed networks involved in encoding, storage, "
            "and retrieval. The interpretation of these measurements requires careful "
            "attention to experimental design and spatial resolution."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Neural representations of visual information in the cortex",
        "abstract": (
            "Visual information is represented across multiple cortical regions that "
            "process features such as orientation, shape, color, and motion. Neural "
            "recordings and functional imaging studies show that these representations "
            "are distributed and transformed across the visual system."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Sleep and memory consolidation in the human brain",
        "abstract": (
            "Sleep supports the consolidation of newly acquired memories. Neural "
            "oscillations during sleep, including slow waves and sleep spindles, are "
            "associated with communication between brain regions involved in memory. "
            "Electrophysiological and imaging methods provide complementary evidence "
            "for these processes."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Neural oscillations and communication between brain regions",
        "abstract": (
            "Oscillatory activity is a common feature of neural systems and may support "
            "communication between distant brain regions. Measures of phase synchrony, "
            "coherence, and cross-frequency coupling are often used to study interactions "
            "between neural populations."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Connectivity analysis of resting-state fMRI data",
        "abstract": (
            "Resting-state functional magnetic resonance imaging measures spontaneous "
            "fluctuations in blood oxygenation. Functional connectivity analysis uses "
            "statistical relationships between regions to characterize large-scale brain "
            "networks. These methods are widely used in studies of development, disease, "
            "and individual differences."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Brain network changes associated with neurological disease",
        "abstract": (
            "Neurological diseases can alter communication between brain regions even "
            "when local activity changes are difficult to detect. Network-based analyses "
            "of neuroimaging data can identify changes in connectivity and may provide "
            "biomarkers for diagnosis, prognosis, or treatment monitoring."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Preprocessing pipelines for electroencephalography",
        "abstract": (
            "EEG preprocessing commonly includes filtering, removal of artifacts, "
            "re-referencing, segmentation, and rejection of noisy trials. Choices made "
            "during preprocessing can affect downstream estimates of spectral power, "
            "connectivity, and classification performance."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Artifact removal in EEG recordings",
        "abstract": (
            "Eye movements, muscle activity, electrode movement, and electrical "
            "interference can contaminate EEG recordings. Automated and semi-automated "
            "methods use statistical properties of the signal to identify and reduce "
            "artifacts while preserving neural activity."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Transfer learning for cross-subject EEG classification",
        "abstract": (
            "EEG-based machine learning models often lose accuracy when applied to a "
            "new participant. Transfer learning methods attempt to reduce this problem "
            "by adapting representations learned from existing subjects to a new subject. "
            "This approach may reduce calibration time for brain-computer interfaces."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Neural decoding of motor imagery",
        "abstract": (
            "Motor imagery involves imagining a movement without physically performing "
            "it. Changes in sensorimotor rhythms can be measured with EEG and used to "
            "decode imagined actions. Successful decoding depends on signal quality, "
            "subject training, and the choice of features and classifier."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Brain-computer interfaces for assistive communication",
        "abstract": (
            "Brain-computer interfaces may help people with severe motor impairments "
            "communicate or control assistive devices. Systems based on EEG, intracortical "
            "recordings, or other neural signals translate brain activity into commands. "
            "Robustness, usability, and user training remain important challenges."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Attention and working memory in human neuroimaging",
        "abstract": (
            "Attention and working memory depend on interactions among distributed brain "
            "networks. Neuroimaging studies investigate how sensory information is "
            "selected, maintained, and updated. Neural activity patterns can provide "
            "information about task demands and individual performance."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Reinforcement learning and decision-making in the brain",
        "abstract": (
            "Reinforcement learning describes how organisms use rewards and prediction "
            "errors to improve future decisions. Computational models and neuroimaging "
            "experiments have linked learning signals to activity in dopaminergic and "
            "frontostriatal circuits."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Single-cell approaches to studying neural diversity",
        "abstract": (
            "The nervous system contains many neuronal and non-neuronal cell types with "
            "distinct molecular and physiological properties. Single-cell sequencing and "
            "spatial transcriptomics allow researchers to characterize cellular diversity "
            "and investigate how cell populations are organized across brain regions."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Spatial transcriptomics in neuroscience",
        "abstract": (
            "Spatial transcriptomics measures gene expression while preserving information "
            "about the location of cells within tissue. In neuroscience, these methods "
            "help connect molecular cell types with anatomical organization and disease "
            "processes."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Neuroinflammation and microglial activation",
        "abstract": (
            "Microglia are immune cells in the central nervous system that respond to "
            "injury, infection, and changes in the neural environment. Their activation "
            "can influence synaptic function and neuronal survival. Neuroinflammation "
            "has been studied in aging and several neurological disorders."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Protein aggregation in neurodegenerative disease",
        "abstract": (
            "Abnormal protein aggregation is a feature of several neurodegenerative "
            "diseases. Misfolded proteins can disrupt cellular transport, synaptic "
            "function, and neuronal survival. Understanding aggregation mechanisms may "
            "support the development of diagnostic markers and therapeutic strategies."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
    {
        "title": "Early detection of neurodegenerative disease using biomarkers",
        "abstract": (
            "Biomarkers from imaging, cerebrospinal fluid, blood, and digital assessments "
            "may help detect neurodegenerative disease before substantial clinical "
            "symptoms appear. Machine learning can combine multiple measurements, but "
            "validation across independent cohorts is necessary to assess generalization."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/",
    },
]


def main() -> None:
    output_path = Path(__file__).parent / "neuro_abstracts.csv"
    dataframe = pd.DataFrame(ABSTRACTS)
    dataframe.to_csv(output_path, index=False)

    print(f"Saved {len(dataframe)} abstracts to: {output_path}")
    print(f"Columns: {list(dataframe.columns)}")
    print(f"Missing values: {dataframe.isna().sum().to_dict()}")
    print("\nFirst two titles:")
    for title in dataframe["title"].head(2):
        print(f"- {title}")


if __name__ == "__main__":
    main()