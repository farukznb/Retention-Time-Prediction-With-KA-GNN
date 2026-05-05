# Data Directory

## Contents
- raw/: Place raw CSV or data files here.
- processed/: Place processed PyTorch tensors or files here.

## Instructions
1. Download the METLIN SMRT dataset from <insert-link>.
2. Save raw data in the raw/ directory.
3. Run preprocessing scripts to generate processed data.

## More details
The empirical foundation of this work is the METLIN Small Molecule Retention Time (SMRT) dataset, comprising 80,038 experimentally measured RT values under standardized reverse-phase conditions. The dataset was accessed via official METLIN resources and includes structural information in multiple formats: CSV files with retention times and metadata, SDF files containing 3D molecular structures, and pre-computed 2048-bit Extended Connectivity Fingerprints (ECFP4). The SMRT dataset represents broad chemical diversity, with organoheterocyclic compounds constituting98.2\% of the dataset, benzenoids comprising 24.7\%, organic acids and derivatives accounting for 6.6\%, and other classes including lipids and lignans making up the remaining 4.8\%. Retention times range from near-immediate elution to 2000 seconds, with a median of 415 seconds and mean of 790.11 seconds (standard deviation 206.65 seconds), ensuring that models are trained across varying degrees of molecular hydrophobicity and interaction strength.

We implemented a standardized preprocessing pipeline using RDKit (v2025.9.3) to ensure data quality and consistency. All molecular representations were standardized to unique canonical SMILES strings, followed by cleanup operations that removed salts, solvents, and counterions while normalizing tautomers and preserving stereochemical centers. Quality filtering restricted the dataset to molecules with molecular weight between 150 and 700 Da and heavy atom count between 10 and 50 atoms, achieving a 98.7\% parsing success rate. Duplicates were eliminated based on InChIKey matching to ensure that each unique chemical structure appeared only once in the dataset.

Three complementary molecular representations were generated to capture different aspects of molecular structure relevant to chromatographic retention. Fingerprint representations utilized 1024-bit ECFP4 fingerprints generated using Morgan circular fingerprints with radius 2, capturing local structural environments in a fixed-length binary vector. Molecular descriptors comprised 32 curated features spanning key physicochemical domains including size properties such as molecular weight and heavy atom count, hydrophobicity and polarity measures including MolLogP and Topological Polar Surface Area, molecular flexibility indicators such as rotatable bond count and fraction of sp3 carbons, hydrogen bonding capacity through donor and acceptor counts, topological indices including BertzCT and Balaban's J, and electronic properties such as partial charge descriptors and valence electron counts. 

Two graph-based molecular representations were generated,  one per model family. For the \textbf{KA-GNN family}  (Experiments~1 and~2), each atom node was encoded with a  133-dimensional feature vector and each bond edge with a 14-dimensional feature vector, following the original  KA-GNN implementation~\cite{liu2024kagnn}, capturing  atomic number, degree, formal charge, hybridization,  aromaticity, implicit hydrogen count, chirality, ring  membership, and additional structural descriptors. For  the \textbf{GCN-based GNN} (Experiment~3, MolGCN), each  atom node was encoded with a 23-dimensional feature vector:  atomic type one-hot encoding (C, N, O, S, P, F, Cl, Br, I  $+$ other: 10), hybridisation one-hot (SP, SP2, SP3, SP3D,  SP3D2 $+$ other: 6), aromaticity (1), formal charge (1),  implicit-H count capped at 4 (1), ring membership (1),  and chirality one-hot (CW, CCW $+$ other: 3). Both graph  representations use undirected edges, with bonds represented in both directions in the edge index. The dataset was partitioned using stratified random splitting to ensure proportional representation of chemical classes across training, validation, and test sets. The final split allocated 55,967 compounds (70\%) to training, 11,994 compounds (15\%) to validation, and 11,994 compounds (15\%) to testing. Retention times were normalized using RobustScaler, which centers values at the median and scales by the interquartile range, thereby mitigating the influence of extreme chromatographic outliers that could distort gradient-based optimization. This partitioning constitutes the data split used in Experiment 1 and in the Experiment 3.

In Experiment 2, the dataset was partitioned using stratified three-fold cross-validation, with folds stratified by chemical class to preserve class proportions across all splits and stratified random splitting to ensure proportional representation of chemical classes across training, validation, and test sets. The final split allocated 55,967 compounds (70\%) to training, 11,994 compounds (15\%) to validation, and 11,994 compounds (15\%) to testing. Retention times were normalized using RobustScaler, which centers values at the median and scales by the interquartile range, thereby mitigating the influence of extreme chromatographic outliers that could distort gradient-based optimization. This partitioning constitutes the data split used in Experiment 1 and in the Experiment 3.


\begin{figure}[htbp]
\centering
\includegraphics[width=0.4\textwidth]{data/eda_01_class_imbalance.png}
\caption{Chemical class distribution of the SMRT dataset after preprocessing
($n = 79{,}955$). Organoheterocyclic compounds dominate with 98.2\% of all
entries (78{,}543 compounds), followed by Other unclassified compounds
(699, 0.9\%), Organic Acids \& Amino Acids (467, 0.6\%), Lipids (188,
0.2\%), and rare minority classes including Benzenoids (24), Aliphatic
Organics (18), and Carbohydrates (16), each representing less than 0.1\%
of the dataset. This severe class imbalance means that global metrics are
effectively majority-class metrics, motivating the class-wise stratified
evaluation conducted in Experiment~2.}
\label{fig:eda_class_imbalance}
\end{figure}
\FloatBarrier

\begin{figure}[htbp]
\centering
\includegraphics[width=0.4\textwidth]{data/eda_02_rt_per_class.png}
\caption{Retention time distribution per chemical class. Carbohydrates and
Aliphatic Organics exhibit substantially higher median retention times and
wider spread compared to Organoheterocyclics and Benzenoids, reflecting
differences in hydrophobicity and polarity across compound families. These
distributional differences explain the class-differential error patterns
observed across architectures in Experiment~2.}
\label{fig:eda_rt_per_class}
\end{figure}
\FloatBarrier

\begin{figure}[htbp]
\centering
\includegraphics[width=0.4\textwidth]{data/eda_03_descriptor_kde.png}
\caption{Kernel density estimates of key physicochemical descriptors
(MolLogP, Topological Polar Surface Area, Molecular Weight) across the
SMRT dataset. The multimodal distributions confirm that the dataset spans
a broad and chemically diverse physicochemical space, supporting the use
of both graph-based and descriptor-based modelling components in the
proposed hybrid architectures.}
\label{fig:eda_descriptor_kde}
\end{figure}

