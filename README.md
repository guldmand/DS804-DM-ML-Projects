# DS804-DM-ML-Projects
Jannik, Mathias & Leila

## Project part I: Clustering
<a href="assignment/exercise11-project1.pdf" target="_blank">Assignment</a> <br>
<a href="clustering-with-weka.md" target="_blank">Weka</a> <br>
<a href="clustering-with-elki.md" target="_blank">Elki</a> <br>
<a href="clustering-with-orange.md" target="_blank">Orange</a> <br>
Code <br>
Præsentation <br>

<table>
  <thead>
    <tr>
      <th>Clustering Algorithms</th>
      <th>Beskrivelse</th>
      <th>Formel</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>Partitional Clustering</strong></td><td></td><td></td></tr>
    <tr><td>k-means (MacQueen 1967, Forgy 1965, Lloyd 1982)</td><td>Opdeler data i k klynger ved at minimere variansen inden for hver klynge.</td><td>Minimerer: \\( \\sum_{i=1}^{k} \\sum_{x \\in C_i} \\|x - \\mu_i\\|^2 \\)</td></tr>
    <tr><td>k-medoids (PAM)</td><td>Ligner k-means, men bruger faktiske datapunkter som centre (medoids).</td><td>Minimerer samlet afstand til medoids.</td></tr>
    <tr><td>k-means++ (Arthur og Vassilvitskii, 2007)</td><td>Smart initiering af k-means for bedre klynger og hurtigere konvergens.</td><td>Samme formel som k-means, men med bedre valg af startcentre.</td></tr>

    <tr><td><strong>Density-Based Clustering</strong></td><td></td><td></td></tr>
    <tr><td>DBSCAN (Ester et al., 1996)</td><td>Finder tætte områder baseret på antal naboer indenfor en radius (eps).</td><td>Brug af eps-nabolag og minimumspunkter.</td></tr>
    <tr><td>DBSCAN* (Campello et al., 2013a, 2015)</td><td>Forbedring af DBSCAN for at opdage varierende tæthed.</td><td>-</td></tr>
    <tr><td>OPTICS</td><td>Bevarer klynge-struktur for varierende tæthed, kræver ikke fast eps.</td><td>-</td></tr>
    <tr><td>HDBSCAN</td><td>Hierarkisk version af DBSCAN med stabilitetsscore for klynger.</td><td>-</td></tr>
    <tr><td>Originalt SNN (Shared Nearest Neighbor Clustering) (Jarvis og Patrick, 1973)</td><td>Bygger klynger baseret på delte naboer i stedet for afstande alene.</td><td>Antal fælles naboer mellem punkter.</td></tr>
    <tr><td>SNN-variant (Ertöz et al., 2003)</td><td>Optimeret variant til store datasæt med varierende tæthed.</td><td>-</td></tr>

    <tr><td><strong>Hierarchical Clustering</strong></td><td></td><td></td></tr>
    <tr><td>Agglomerative Clustering (bottom-up)</td><td>Starter med hvert punkt som egen klynge og fusionerer trin for trin.</td><td>Afstandsmål: single, complete eller average linkage.</td></tr>
    <tr><td>Divisive Clustering (top-down)</td><td>Starter med alle punkter samlet og splitter trin for trin.</td><td>-</td></tr>
    <tr><td>BIRCH</td><td>Effektiv clustering af store datasæt vha. CF-træstruktur.</td><td>-</td></tr>

    <tr><td><strong>Fuzzy Clustering</strong></td><td></td><td></td></tr>
    <tr><td>Fuzzy c-means</td><td>Hvert punkt kan tilhøre flere klynger med en tilhørselsgrad.</td><td>Minimerer: \\( \\sum_{i=1}^{n} \\sum_{j=1}^{k} u_{ij}^m \\|x_i - c_j\\|^2 \\)</td></tr>
    <tr><td>Gustafson-Kessel</td><td>Udvider Fuzzy c-means ved at tillade ellipsoide klynger.</td><td>-</td></tr>

    <tr><td><strong>Graph-Theoretic Clustering</strong></td><td></td><td></td></tr>
    <tr><td>Spectral Clustering</td><td>Bruger eigenvektorer af Laplacian matrix til at finde klynger.</td><td>Bygger på Laplacian: \\( L = D - A \\)</td></tr>
    <tr><td>Minimum Spanning Tree (MST) Clustering</td><td>Bygger MST og fjerner lange kanter for at identificere klynger.</td><td>-</td></tr>

    <tr><td><strong>Neural Net-Based Clustering</strong></td><td></td><td></td></tr>
    <tr><td>Self-Organizing Maps (SOM)</td><td>Bruger neurale netværk til at projektere data til lavdimensionelle kort.</td><td>-</td></tr>
    <tr><td>Deep Embedded Clustering (DEC)</td><td>Kombinerer deep learning feature learning og clustering.</td><td>-</td></tr>
  </tbody>
</table>






## Project part II: Classification
Assignment <br>