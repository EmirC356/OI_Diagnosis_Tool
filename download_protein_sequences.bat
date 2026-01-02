@echo off
REM Download COL1A1 and COL1A2 protein sequences from UniProt

echo Downloading COL1A1 protein sequence...
curl -o COL1A1_protein.fasta "https://www.uniprot.org/uniprot/P02452.fasta"

echo Downloading COL1A2 protein sequence...
curl -o COL1A2_protein.fasta "https://www.uniprot.org/uniprot/P08123.fasta"

echo Download complete!
echo Files created:
echo   - COL1A1_protein.fasta
echo   - COL1A2_protein.fasta
pause
