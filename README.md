# Osteogenesis Imperfecta Diagnosis Model
**Machine Learning-Based Variant Pathogenicity Prediction for COL1A1 and COL1A2 Genes**

## Project Overview
This project develops disease-specific machine learning models to predict the pathogenicity of genetic variants in COL1A1 and COL1A2 genes associated with Osteogenesis Imperfecta (brittle bone disease). The models achieve 97.2% accuracy, outperforming industry-standard tools like REVEL by over 7%.

## Key Results
- **Best Model**: Random Forest (97.2% test accuracy)
- **Performance vs REVEL**: +7.2% accuracy, +13.8% average improvement across all metrics
- **Dataset**: 3,105 variants (1,682 pathogenic, 1,423 benign) from ClinVar
- **Models Tested**: Logistic Regression, Random Forest, SVM, Gradient Boosting

## Project Structure

```
ENS210_Project/
│
├── data/                           # Raw data files (not tracked in git)
│
├── 01_data_cleaning/               # Data cleaning and preprocessing
│   ├── Data_Cleaning.jpynb         # Main data cleaning notebook
│   ├── Control.jpynb               # Quality control checks
│   ├── FastFileEditor.ipynb        # Utility for file editing
│   └── cleaned_COL1A1.csv          # Cleaned dataset
│
├── 02_data_exploration/            # Exploratory data analysis
│   ├── 01_data_exploration.py      # EDA script
│   └── data_exploration_plots.png  # Visualization of variant distributions
│
├── 03_feature_engineering/         # Feature extraction and engineering
│   └── 02_feature_engineering.py   # Feature engineering pipeline
│
├── 04_models/                      # Machine learning models
│   ├── 03_ml_models.py             # ML training and evaluation script
│   ├── Gene_mutation_project.ipynb # Model development notebook
│   ├── model_comparison.csv        # Performance metrics comparison
│   ├── model_evaluation.png        # Model performance visualizations
│   └── feature_importance.csv      # Feature importance rankings
│
├── 05_tool_comparison/             # Comparison with existing tools
│   ├── 05_tool_comparison_analysis.py           # Tool comparison analysis
│   ├── 06a_prepare_variants_for_tools.py        # Variant preparation
│   ├── 06b_query_dbnsfp.py                      # dbNSFP querying
│   ├── 06c_use_ensembl_vep.py                   # VEP API usage
│   ├── 07_create_test_files_for_revel.py        # REVEL test file generation
│   ├── 08_create_vep_input_file.py              # VEP input formatting
│   ├── 09_create_sift_input_files.py            # SIFT input (initial)
│   ├── 10_fix_sift_format.py                    # SIFT format fix (v2)
│   ├── 11_create_correct_sift_format.py         # SIFT format (final)
│   ├── 12_parse_vep_results.py                  # VEP results parser
│   ├── REVEL_Confusion_Matrix_code.py           # REVEL comparison
│   │
│   ├── test_variants_for_revel.tsv              # Test set with labels
│   ├── complete_test_set.tsv                    # Full test set (621 variants)
│   ├── missense_test_set.tsv                    # Missense only (176 variants)
│   ├── missense_variants_for_tools.tsv          # Tool query format
│   │
│   ├── test_variants_vep.vcf                    # VEP input (VCF format)
│   ├── test_variants_vep_default.txt            # VEP input (default format)
│   ├── test_variants_vep_hgvs.txt               # VEP input (HGVS format)
│   ├── test_variants_vep_rsid.txt               # VEP input (rsID format)
│   │
│   ├── COL1A1_protein.fasta                     # UniProt P02452
│   ├── COL1A2_protein.fasta                     # UniProt P08123
│   ├── sift_COL1A1_substitutions_correct.txt    # SIFT input (X#Y format)
│   ├── sift_COL1A2_substitutions_correct.txt    # SIFT input (X#Y format)
│   │
│   └── 1eXf8ZzkWueyMwc6.txt                     # VEP results file
│
├── 06_results/                     # Analysis results and visualizations
│   ├── confusion_matrices.png                   # Model confusion matrices
│   ├── tool_comparison_comprehensive.png        # Multi-metric comparison
│   ├── tool_comparison_summary.txt              # Summary statistics
│   ├── tool_performance_comparison.csv          # Performance metrics
│   ├── all_variants_with_tool_predictions_vep.tsv
│   ├── missense_vep_predictions.tsv
│   └── variant_predictions_with_consensus.tsv
│
├── 07_documentation/               # Project documentation
│   ├── COMPREHENSIVE_PROJECT_REPORT.md          # Full project report
│   ├── DETAILED_METHODOLOGY_EXPLANATION.md      # Methods documentation
│   ├── PROJECT_SUMMARY.md                       # Project summary
│   ├── PROJECT_SUMMARY_SHORT.md                 # Brief summary
│   ├── 04_existing_tools_guide.md               # Guide to existing tools
│   ├── 06_get_existing_tool_predictions.md      # Tool query instructions
│   ├── INSTRUCTIONS_FOR_TOOL_QUERIES.txt
│   ├── TEST_SET_INSTRUCTIONS.txt
│   ├── READY_FOR_VEP_SUBMISSION.md
│   ├── VEP_SUBMISSION_GUIDE.txt
│   ├── SIFT_SUBMISSION_GUIDE.txt
│   ├── VARIANT_PREPARATION_SUMMARY.txt
│   ├── TOOL_COMPARISON_RESULTS.md
│   └── TOOL_PREDICTIONS_STATUS.md
│
├── 08_presentation/                # Presentation materials
│   ├── 34110_Emir_Ceylan_ENS210_Presentation.pdf
│   └── presentation_talking_notes.md            # 3-minute talk notes
│
├── download_protein_sequences.bat  # Utility script for Windows
├── download_protein_sequences.sh   # Utility script for Linux/Mac
└── README.md                       # This file
```

## Workflow

### 1. Data Cleaning ([01_data_cleaning/](01_data_cleaning/))
- Downloaded variants from ClinVar for COL1A1 and COL1A2 genes
- Filtered out Variants of Uncertain Significance (VUS)
- Excluded non-OI related variants and complex structural variants (>50 bp)
- Final dataset: 3,105 variants (54.2% pathogenic, 45.8% benign)

### 2. Data Exploration ([02_data_exploration/](02_data_exploration/))
- Analyzed variant type distributions
- Examined molecular consequence patterns
- Visualized class balance and pathogenicity by variant type
- Key finding: Missense variants show balanced pathogenicity

### 3. Feature Engineering ([03_feature_engineering/](03_feature_engineering/))
- Extracted genomic features (position, chromosome, gene)
- Encoded variant types and molecular consequences
- Created protein-level features (amino acid changes, glycine substitutions)
- Engineered domain-specific features for collagen structure

### 4. Model Development ([04_models/](04_models/))
- Trained 4 ML models with 5-fold stratified cross-validation
- Models: Logistic Regression, Random Forest, SVM, Gradient Boosting
- Best performance: Random Forest (97.2% test accuracy)
- Minimal overfitting: <2% gap between train and test accuracy

### 5. Tool Comparison ([05_tool_comparison/](05_tool_comparison/))
- Prepared test variants for external tools (REVEL, CADD, SIFT, PolyPhen-2)
- Submitted variants to Ensembl VEP and SIFT web interfaces
- Parsed results and compared with ML model performance
- Result: ML models outperform generic tools by 7-18% across metrics

### 6. Results Analysis ([06_results/](06_results/))
- Generated comprehensive performance comparisons
- Created visualizations for model evaluation
- Documented improvement over existing tools
- Key metric: +13.8% average improvement in accuracy over REVEL

## Key Findings

1. **Disease-specific models outperform generic tools**: By incorporating domain knowledge about collagen structure and OI-specific features, ML models achieved 97.2% accuracy vs 90% for REVEL

2. **Minimal overfitting**: All models showed robust generalization with <2% train-test accuracy gap

3. **Feature importance**: Glycine substitutions, variant position, and molecular consequence type were the most predictive features

4. **Clinical relevance**: The model can assist in interpreting VUS and prioritizing variants for functional validation

## Future Directions

- Integrate 3D protein folding data from AlphaFold to model structural impact
- Expand to the 15+ rarer genes associated with OI
- Develop web interface for clinical variant interpretation
- Validate on independent clinical cohorts

## Requirements

```bash
# Python packages
pandas
numpy
scikit-learn
matplotlib
seaborn
biopython

# For Jupyter notebooks
jupyter
ipykernel
```

## Usage

### Running the complete pipeline:

```bash
# 1. Data cleaning
python 01_data_cleaning/Data_Cleaning.jpynb

# 2. Data exploration
python 02_data_exploration/01_data_exploration.py

# 3. Feature engineering
python 03_feature_engineering/02_feature_engineering.py

# 4. Train models
python 04_models/03_ml_models.py

# 5. Compare with tools
python 05_tool_comparison/05_tool_comparison_analysis.py
```

### Preparing variants for external tools:

```bash
# VEP submission
python 05_tool_comparison/08_create_vep_input_file.py

# SIFT submission
python 05_tool_comparison/11_create_correct_sift_format.py

# Parse VEP results
python 05_tool_comparison/12_parse_vep_results.py
```

## Data Sources

- **Variant data**: ClinVar (https://www.ncbi.nlm.nih.gov/clinvar/)
- **Protein sequences**: UniProt (COL1A1: P02452, COL1A2: P08123)
- **Genome reference**: GRCh38
- **Tool comparisons**: Ensembl VEP, SIFT, PolyPhen-2, REVEL, CADD

## Citation

If you use this work, please cite:
```
Ceylan, E. (2025). Machine Learning-Based Pathogenicity Prediction for
Osteogenesis Imperfecta Variants. ENS 210 Course Project.
```

## License

This project is for educational purposes as part of the ENS 210 course.

## Contact

Emir Ceylan (Student ID: 34110)
Course: ENS 210
Date: December 31, 2025

---

**Note**: The `data/` folder contains raw ClinVar downloads and is not tracked in version control due to file size. Download instructions are provided in [07_documentation/](07_documentation/).
