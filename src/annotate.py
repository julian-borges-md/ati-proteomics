"""
annotate.py — Biological pathway annotation of top predictive proteins
=======================================================================
RO-2026-001 | Frontier Translational Research Lab

Maps SomaScan feature identifiers to protein names and assigns pathway
categories per RO Section 3.6. Run after train.py.

Usage:
    python3 src/annotate.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR  = os.path.join(BASE_DIR, "artifacts")

fi = pd.read_csv(os.path.join(ART_DIR, "feature_importance.csv"))

# Pathway assignments — representative ATI proteins from Schmidt et al. 2024
# Replace mapping keys with actual SomaScan aptamer IDs from real data.
KNOWN = {
    "KIM1":     ("tubular injury markers",
                 "KIM-1/TIM-1; proximal tubule injury sensor"),
    "NGAL":     ("tubular injury markers",
                 "Lipocalin-2; AKI biomarker, proximal tubule"),
    "CYSC":     ("tubular injury markers",
                 "Cystatin C; GFR estimation, tubular reabsorption"),
    "UMOD":     ("filtration and barrier function",
                 "Uromodulin; thick ascending limb, barrier integrity"),
    "IL18":     ("immune and inflammatory mediators",
                 "IL-18; inflammasome activation in tubular cells"),
    "CXCL10":   ("immune and inflammatory mediators",
                 "IP-10; T-cell chemokine in kidney injury"),
    "CCL2":     ("immune and inflammatory mediators",
                 "MCP-1; macrophage recruitment, tubulointerstitial inflammation"),
    "TNF":      ("immune and inflammatory mediators",
                 "TNF-alpha; pro-inflammatory cytokine in AKI"),
    "IL6":      ("immune and inflammatory mediators",
                 "IL-6; systemic inflammation, tubular stress"),
    "SOD1":     ("oxidative stress response",
                 "SOD1; ROS scavenging in tubular epithelia"),
    "GPX1":     ("oxidative stress response",
                 "GPX1; glutathione peroxidase, oxidative damage protection"),
    "HMOX1":    ("oxidative stress response",
                 "HO-1; cytoprotective response to oxidative stress"),
    "CASP3":    ("apoptosis and cell death",
                 "Caspase-3; executor caspase in tubular apoptosis"),
    "BCL2":     ("apoptosis and cell death",
                 "BCL-2; anti-apoptotic regulator in tubular stress"),
    "EGF":      ("tissue repair and regeneration",
                 "EGF; tubular regeneration and repair signaling"),
    "HGF":      ("tissue repair and regeneration",
                 "HGF; hepatocyte growth factor, tubular repair"),
    "PODOCIN":  ("filtration and barrier function",
                 "Podocin; slit diaphragm structural protein"),
    "NEPHRIN":  ("filtration and barrier function",
                 "Nephrin; glomerular filtration barrier integrity"),
}

rows = []
for _, row in fi.head(min(20, len(fi))).iterrows():
    feat = row["feature"]
    matched = False
    for key, (pathway, desc) in KNOWN.items():
        if key.lower() in feat.lower():
            rows.append({"rank": int(row["rank"]), "feature": feat,
                         "protein_name": key, "pathway_category": pathway,
                         "functional_description": desc,
                         "source": "Literature + UniProt"})
            matched = True
            break
    if not matched:
        rows.append({"rank": int(row["rank"]), "feature": feat,
                     "protein_name": feat,
                     "pathway_category": "other",
                     "functional_description": "Requires SomaScan manifest mapping",
                     "source": "PLACEHOLDER"})

annot = pd.DataFrame(rows)
annot.to_csv(os.path.join(ART_DIR, "protein_annotation.csv"), index=False)
print("Protein annotation saved.")
print(annot[["rank", "feature", "pathway_category"]].to_string(index=False))
