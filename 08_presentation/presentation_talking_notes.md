# 3-Minute Presentation Talking Notes
## Osteogenesis Imperfecta Diagnosis Model

---

### SLIDE 1: Title Slide
*No speech needed - just introduce yourself*

---

### SLIDE 2: Osteogenesis Imperfecta Background

**[30 seconds]**

"Osteogenesis Imperfecta, or brittle bone disease, is a genetic disorder caused by mutations in the COL1A1 and COL1A2 genes. These genes encode type I collagen - the most abundant structural protein in our bones and connective tissues. Understanding how mutations in these genes lead to disease is critical for diagnosis and treatment."

---

### SLIDE 3: Type I Collagen Structure

**[20 seconds]**

"Here's why these mutations matter: Type I collagen forms a triple helix structure with a critical Gly-X-Y pattern. Glycine - the smallest amino acid - must occupy every third position. Even a single glycine substitution can destabilize the entire triple helix, leading to bone fragility."

---

### SLIDE 4: Data Configuration

**[25 seconds]**

"For this project, I obtained 3,584 variants from ClinVar. After excluding variants of uncertain significance and non-OI related variants, I had 3,105 clean variants - 1,682 pathogenic and 1,423 benign. This represents a fairly balanced dataset split between COL1A1 and COL1A2 genes."

---

### SLIDE 5: Data Exploration
*Skip - visualization speaks for itself*

---

### SLIDE 6: Methods and Results

**[40 seconds]**

"I trained four machine learning models: Logistic Regression as a baseline, Random Forest for handling non-linear relationships, Support Vector Machine for high-dimensional classification, and Gradient Boosting for optimal performance.

All models achieved impressive results - between 96.7% and 97.2% test accuracy. The key finding here is the minimal overfitting - all models showed less than 2% gap between training and test accuracy, indicating robust generalization."

---

### SLIDE 7: Tool Comparison

**[45 seconds]**

"Now here's where it gets interesting. I compared my models against existing industry-standard tools like REVEL, CADD, PolyPhen-2, and SIFT.

My Random Forest model achieved 97.2% accuracy, while REVEL - the current gold standard - only reached 90%. That's over 7% improvement across all metrics. You can see in the precision-recall trade-off plot that our disease-specific models consistently outperform generic tools. On average, we achieved 13.8% improvement in accuracy, 18.1% in ROC-AUC, and significant gains in precision and recall."

---

### SLIDE 8: Future Directions

**[20 seconds]**

"While these results are impressive, there's still room for improvement. Future work could integrate 3D protein folding data from AlphaFold to understand how mutations physically distort the collagen structure, and expand the model to cover the 15+ rarer genes associated with OI."

---

### SLIDE 9: Conclusion

**[20 seconds]**

"The key takeaway: disease-specific machine learning models that incorporate unique biological knowledge about the disease significantly outperform generalized prediction tools. By combining domain expertise with machine learning, we can build more accurate diagnostic tools that ultimately lead to better patient outcomes."

---

### SLIDE 10: Thank You
*No speech needed - take questions*

---

## TIMING BREAKDOWN:
- Slide 2: 30s
- Slide 3: 20s
- Slide 4: 25s
- Slide 6: 40s
- Slide 7: 45s
- Slide 8: 20s
- Slide 9: 20s

**TOTAL: ~3 minutes**

---

## TIPS FOR DELIVERY:
1. Speak at moderate pace - don't rush the comparison slide (7)
2. Emphasize the "7% improvement" and "over 13% average improvement" numbers
3. Make eye contact when stating the conclusion about disease-specific approaches
4. Use hand gestures to point at the triple helix structure on slide 3
5. Pause briefly before revealing the comparison results on slide 7 for impact
