# DS804-DM-ML-Projects
Jannik, Mathias & Leila

## Project part I: Clustering
<a href="assignment/exercise11-project1.pdf" target="_blank">Assignment</a> <br>
<a href="clustering-with-weka.md" target="_blank">Weka</a> <br>
<a href="clustering-with-elki.md" target="_blank">Elki</a> <br>
<a href="clustering-with-orange.md" target="_blank">Orange</a> <br>
Code <br>
Præsentation <br>

| Clustering Algorithm | Beskrivelse | Formel |
|---|---|---|
| <br> | <br> | <br> |
| **Partitional Clustering** |  |  |
| **k-means** (MacQueen 1967, Forgy 1965, Lloyd 1982) | Opdeler data i k klynger ved at minimere variansen inden for hver klynge. | Minimerer: $\\sum_{i=1}^{k} \\sum_{x \\in C_i} \\|x - \\mu_i\\|^2$ |
| **k-medoids** (PAM) | Ligner k-means, men bruger faktiske datapunkter som centre (medoids). | Minimerer samlet afstand til medoids |
| **k-means++** (Arthur og Vassilvitskii, 2007) | Smart initiering af k-means for bedre klynger og hurtigere konvergens. | Samme som k-means, men bedre initiering |
| <br> | <br> | <br> |
| **Density-Based Clustering** |  |  |
| **DBSCAN** (Ester et al., 1996) | Finder tætte områder baseret på antal naboer indenfor en radius (eps). | eps-nabolag og minimum antal punkter |
| **DBSCAN*** (Campello et al., 2013a, 2015) | Forbedring af DBSCAN for varierende tæthed. | - |
| **OPTICS** | Bevarer klynge-struktur for varierende tæthed, kræver ikke fast eps. | - |
| **HDBSCAN** | Hierarkisk version af DBSCAN med stabilitetsmåling. | - |
| **SNN** (Jarvis og Patrick, 1973) | Bygger klynger baseret på delte naboer i stedet for afstand alene. | Antal fælles naboer |
| **SNN-variant** (Ertöz et al., 2003) | Optimeret variant til store datasæt med varierende tæthed. | - |
| <br> | <br> | <br> |
| **Hierarchical Clustering** |  |  |
| **Agglomerative Clustering** (bottom-up) | Starter med hvert punkt som egen klynge og fusionerer trin for trin. | Afstandsmål: single, complete eller average linkage |
| **Divisive Clustering** (top-down) | Starter med alle punkter samlet og splitter trin for trin. | - |
| **BIRCH** | Effektiv clustering af store datasæt vha. CF-træstruktur. | - |
| **Fuzzy Clustering** |  |  |
| **Fuzzy c-means** | Hvert punkt kan tilhøre flere klynger med en tilhørselsgrad. | Minimerer: $\\sum_{i=1}^{n} \\sum_{j=1}^{k} u_{ij}^m \\|x_i - c_j\\|^2$ |
| **Gustafson-Kessel** | Tillader ellipsoide klynger, udvider Fuzzy c-means. | - |
| <br> | <br> | <br> |
| **Graph-Theoretic Clustering** |  |  |
| **Spectral Clustering** | Bruger eigenvektorer af Laplacian matrix til at finde klynger. | Laplacian: $L = D - A$ |
| **Minimum Spanning Tree** (MST) Clustering | Bygger MST og fjerner lange kanter for at identificere klynger. | - |
| <br> | <br> | <br> |
| **Neural Net-Based Clustering** |  |  |
| **Self-Organizing Maps** (SOM) | Bruger neurale netværk til at projektere data til lavdimensionelle kort. | - |
| **Deep Embedded Clustering** (DEC) | Kombinerer deep learning feature learning og clustering. | - |







## Project part II: Classification
Assignment <br>