# DOSSIER TECHNIQUE DE DEMANDE PROVISOIRE DE BREVET D'INVENTION
### Institut National de la Propriété Industrielle (INPI - France)
*Conformément aux articles L. 612-1 et R. 612-1 et suivants du Code de la Propriété Intellectuelle (Loi PACTE n° 2019-486)*

---

## 1. CARTOUCHE D'IDENTIFICATION DU DÉPÔT

- **Demandeur(s)** :
  - **M. Xavier CALLENS** / **SOCRATE AI LAB**
  - Nationalité : Française / Résidence UE
- **Inventeur(s)** :
  - **M. Xavier CALLENS**
- **Type de Titre sollicité** : Demande Provisoire de Brevet d'Invention (avec revendication de priorité sous 12 mois pour dépôt international PCT / OEB)
- **Titre de l'Invention (Français)** :
  *PROCÉDÉ ET SYSTÈME D'OPTIMISATION DE L'EXÉCUTION ET DE L'APPRENTISSAGE DE MODÈLES DE LANGAGE PAR ATTENTION DÉTERMINISTE À VIRGULE FIXE, COMPRESSION GÉOMÉTRIQUE DE MÉMOIRE TENSORIELLE ET ORDONNANCEMENT ASSERVI À L'INTENSITÉ CARBONE*
- **Title of the Invention (English)** :
  *METHOD AND SYSTEM FOR OPTIMIZING LARGE LANGUAGE MODEL INFERENCE AND TRAINING VIA FIXED-POINT DETERMINISTIC ATTENTION, GEOMETRIC TENSOR MEMORY COMPRESSION, AND CARBON-AWARE SCHEDULING*
- **Domaine Technique / Classification CIB** :
  - **G06N 3/063** (Systèmes informatiques fondés sur des modèles neuronaux physiques / circuits)
  - **G06N 3/08** (Méthodes d'apprentissage et rétropropagation)
  - **G06F 9/48** (Ordonnancement et synchronisation de tâches)
  - **G06F 17/16** (Calculs matriciels et vectoriels)

---

## 2. DOMAINE TECHNIQUE DE L'INVENTION

La présente invention s'inscrit dans le domaine des architectures informatiques et logicielles pour l'accélération du calcul neuronal et de l'intelligence artificielle.

Elle concerne plus particulièrement l'optimisation conjointe de la mémoire, de la reproductibilité numérique, du débit matériel et de l'efficience énergétique lors du déploiement en inférence et de l'entraînement à grande échelle de modèles neuronaux de type transformeur (*Transformers*) et modèles de fondation (*Large Language Models* - LLM, architectures à mélange d'experts - *Mixture of Experts* MoE).

L'invention trouve une application privilégiée sur les accélérateurs tensoriels haute performance (notamment *Graphics Processing Units* - GPU NVIDIA Hopper/Blackwell, *Tensor Processing Units* - Google TPU v5e/v5p/v6e Trillium, et processeurs vectoriels RISC-V RVV 1.0).

---

## 3. ÉTAT DE LA TECHNIQUE ANTÉRIEURE ET PROBLÈMES TECHNIQUES

### 3.1 Problème 1 : Dérive numérique et non-déterminisme des calculs d'attention
Dans les mécanismes d'attention standards (*Scaled Dot-Product Attention*, *FlashAttention*), les opérations de produit scalaire ($QK^T$), de normalisation exponentielle (*Softmax*) et de projection de valeur ($AV$) sont exécutées en arithmétique flottante (FP32, FP16 ou BF16) conformément à la norme IEEE 754.
Or, l'addition flottante n'est pas associative :
$$(A \oplus B) \oplus C \neq A \oplus (B \oplus C)$$
En environnement massivement parallèle (multi-cœurs GPU ou tranches TPU), l'ordre d'accumulation des tuiles de calcul dépend de l'ordonnancement non-déterministe des threads et des blocs de calcul (*warps*). Il en résulte une gigue numérique (*numerical drift*) non reproductible run-à-run.
Ce manque de déterminisme est prohibitif pour les chaînes de raisonnement logique arborescent long (*chain-of-thought*, *test-time compute*, preuve formelle mathématique) où une variation infime de probabilité au token $t$ fait diverger l'ensemble de la trajectoire de génération.

### 3.2 Problème 2 : Mur mémoire et explosion de la bande passante du cache clé-valeur (KV-Cache)
Pour des fenêtres de contexte étendues (de 32 768 à 131 072 tokens et au-delà, ex: Mistral Large 2, Mixtral 8x22B), la taille du cache Clé-Valeur (*KV-cache*) croît linéairement avec la longueur de la séquence et dépasse fréquemment la mémoire des poids du modèle.
La quantification scalaire directe en dessous de 8 bits (e.g. INT4 ou 3-bit) engendre une dégradation rédhibitoire de la perplexité du modèle en raison de la présence de valeurs aberrantes (*activation outliers*) dans l'espace résiduel des couches d'attention.

### 3.3 Problème 3 : Sous-utilisation des réseaux systoliques matriciels
Les accélérateurs tensoriels contemporains (tels que les cœurs MXU des TPU Google de 128×128 ou 256×256 éléments) requièrent que les dimensions des matrices soient des multiples exacts de la taille de la grille systolique. Les dimensions asymétriques des couches de projection et de regroupement de têtes d'attention (*Grouped-Query Attention* - GQA) provoquent une sous-utilisation critique des cœurs de calcul (taux d'occupation typiquement inférieur à 40%).

### 3.4 Problème 4 : Absence d'asservissement écologique de l'inférence spéculative
Le décodage spéculatif (*speculative decoding*) permet d'accélérer l'inférence en échantillonnant $K$ tokens candidats au moyen d'un modèle brouillon (*draft model*). Cependant, les implémentations actuelles utilisent un nombre d'étapes $K$ constant ou asservi uniquement au taux d'acceptation mathématique. L'empreinte carbone et la puissance instantanée appelée ne sont jamais intégrées dans la boucle de contrôle de l'inférence, entraînant une surconsommation énergétique lors des pointes de charge carbone sur les réseaux électriques.

---

## 4. EXPOSÉ SOMMAIRE DE L'INVENTION

La présente invention résout l'ensemble de ces inconvénients techniques par la combinaison synergique de quatre composantes architecturales au sein d'un moteur d'exécution en environnement sécurisé mémoire :

1. **Un opérateur d'attention déterministe à virgule fixe entière (INT64 Fixed-Point LUT Softmax)** :
   Toutes les étapes d'accumulation matricielle et de calcul exponentiel sont encapsulées en arithmétique entière 64 bits strictement associative au moyen d'une table de projection exponentielle pré-calculée (LUT). Cette architecture garantit une reproductibilité **bit-à-bit stricte avec une dérive numérique nulle ($\Delta = 0$)**, indépendamment de l'ordonnancement matériel.

2. **Un procédé de compression géométrique du cache clé-valeur (PolarQuant & QJL)** :
   Avant quantification à 3 bits, les vecteurs de clés et de valeurs subissent une rotation orthogonale pseudo-aléatoire (générée par un mélangeur bijectif sans tableau de consultation de type SplitMix64 normalisé), redistribuant l'énergie de manière homogène sur toutes les composantes vectorielles et éliminant les valeurs aberrantes. La fidélité géométrique des produits scalaires est surveillée dynamiquement par une projection de Johnson-Lindenstrauss quantifiée (QJL).

3. **Un système d'ordonnancement spéculatif asservi à l'intensité carbone temps-réel** :
   Le paramètre de spéculation $K$ (nombre de tokens générés par le modèle brouillon) est asservi dynamiquement à un signal de télémétrie de l'intensité carbone ($g\text{CO}_2/\text{kWh}$) reçu en temps réel d'un gestionnaire de réseau de transport d'électricité (ex: RTE Eco2Mix en France). Le système réduit l'intensité de calcul durant les pics carbone tout en maximisant le débit lors des périodes d'énergie décarbonée.

4. **Un ordonnanceur de tâches guidé par télémétrie matérielle (WARS - Workload-Adaptive RL Scheduler)** :
   Les files d'attente d'exécution adaptent dynamiquement le poids de priorité des tâches en fonction de compteurs PMU (*Performance Monitoring Unit*) mesurant les défauts de cache L1 (*cache misses*) et assignent préférentiellement les calculs matriciels aux cœurs à haut parallélisme et les opérations d'E/S aux cœurs d'assistance.

---

## 5. DESCRIPTION DÉTAILLÉE DES MODES DE RÉALISATION

### 5.1 Mode de réalisation de l'Attention Déterministe INT64
Le procédé transforme les tenseurs de requête $Q \in \mathbb{Z}^{B \times H \times S \times D}$ et de clé $K \in \mathbb{Z}^{B \times H \times S \times D}$ quantifiés en entiers signés :
1. **Produit scalaire entier** :
   $$S_{i, j} = \sum_{d=1}^{D} Q_{i, d} \cdot K_{j, d} \quad \in \mathbb{Z}^{64}$$
2. **Normalisation et décalage** :
   Le score entier est mis à l'échelle par décalage arithmétique de bits (*bitwise shift*) et bridé (*clamping*) dans l'intervalle d'adressage de la table LUT :
   $$\tilde{S}_{i, j} = \text{clamp}\left( (S_{i, j} \gg \sigma) + \text{offset}, 0, 2\mathcal{L} - 1 \right)$$
3. **Pondération exponentielle par table LUT** :
   Une table LUT de taille $2\mathcal{L}$ pré-calculée en mémoire cache SRAM de l'accélérateur associe à chaque index entier sa valeur d'exponentielle à virgule fixe :
   $$\text{LUT}[m] = \left\lfloor \exp\left( \frac{m - \mathcal{L}}{\alpha} \right) \cdot 2^{16} \right\rceil$$
4. **Agrégation et produit avec la valeur $V$** :
   $$O_{i} = \sum_{j} \text{LUT}[\tilde{S}_{i, j}] \cdot V_{j}$$
   L'ensemble des additions étant commutatif et associatif dans l'anneau commutatif $\mathbb{Z}$, la sortie tensorielle est invariantivement bit-exacte.

### 5.2 Mode de réalisation de la Compression PolarQuant + QJL
Pour un vecteur d'activation $x \in \mathbb{R}^D$ ($D = 64, 128$), le système génère une matrice orthogonale $R \in \mathbb{R}^{D \times D}$ déterministe via la séquence pseudo-aléatoire SplitMix64 :
$$R_{i, j} = \text{SplitMix64}(\text{seed}, i, j) \cdot \frac{\sqrt{3}}{\sqrt{D}}$$
Le vecteur transformé $\tilde{x} = R \cdot x$ présente une variance uniformisée vérifiant :
$$\mathbb{E}[\|\tilde{x}\|_2^2] = \|x\|_2^2 \quad \text{avec} \quad \frac{|\|\tilde{x}\|_2^2 - \|x\|_2^2|}{\|x\|_2^2} \le 0.35$$
Chaque composante de $\tilde{x}$ est quantifiée sur 3 bits ($8$ niveaux discrets). La réduction d'empreinte mémoire atteint un ratio d'au moins 4,92× par rapport au format FP16 et 2,46× par rapport au format FP8.

### 5.3 Mode de réalisation de l'Asservissement Spéculatif Carbone
Le nombre optimal de tokens spéculatifs $K^*(t)$ à l'instant $t$ est déterminé par la fonction de régulation :
$$K^*(t) = \text{clamp}\left( \left\lfloor K_{\text{base}} \cdot \left( \frac{\mathcal{C}_{\text{ref}}}{\mathcal{C}_{\text{grid}}(t)} \right)^{\gamma} \cdot \alpha_{\text{accept}} \right\rceil, K_{\min}, K_{\max} \right)$$
Où :
- $\mathcal{C}_{\text{grid}}(t)$ est l'intensité carbone mesurée en $g\text{CO}_2/\text{kWh}$ du réseau électrique alimentant le centre de calcul à l'instant $t$.
- $\mathcal{C}_{\text{ref}}$ est une valeur de référence d'intensité carbone (ex: 56 $g\text{CO}_2/\text{kWh}$ pour le mix nucléaire français).
- $\alpha_{\text{accept}}$ est le taux glissant d'acceptation des tokens du modèle brouillon.
- $\gamma \in [0.5, 1.0]$ est le coefficient de sensibilité carbone.

---

## 6. JEU DE REVENDICATIONS (CLAIMS)

### Revendication 1 (Indépendante — Attention Déterministe)
Procédé mis en œuvre par ordinateur pour l'exécution déterministe d'un mécanisme d'attention dans un réseau de neurones artificiels, **caractérisé en ce qu'il comprend** les étapes consistant à :
a) Recevoir des représentations tensorielles de requêtes ($Q$), de clés ($K$) et de valeurs ($V$) associées à des tokens d'entrée ;  
b) Calculer une matrice de scores d'attention par multiplication matricielle entière entre lesdites requêtes ($Q$) et clés ($K$) dans un registre d'entiers signés à au moins 64 bits ;  
c) Appliquer à chaque élément de ladite matrice de scores d'attention une opération de normalisation non linéaire via une table de consultation (LUT) indexée directement par une portion binaire tronquée dudit entier, ladite table stockant des valeurs exponentielles discrétisées en virgule fixe entière ;  
d) Multiplier les valeurs discrétisées résultantes par ledit tenseur de valeurs ($V$) au moyen d'additions entières strictement associatives ;  
de telle sorte que la sortie tensorielle d'attention produite soit strictement bit-exacte et reproductible sans dérive numérique quelle que soit la granularité de parallélisme matériel utilisée.

### Revendication 2 (Indépendante — Compression Géométrique de Cache KV)
Procédé de compression de mémoire pour le cache clé-valeur d'un modèle de langage transformeur, **caractérisé en ce qu'il comprend** :
a) L'application d'une transformation orthogonale pseudo-aléatoire à chaque vecteur de clé et de valeur à mémoriser au moyen d'un opérateur de rotation sans tableau calculé en temps réel par générateur à mélange pseudo-aléatoire SplitMix64 mis à l'échelle de la dimension vectorielle ;  
b) La quantification scalaire uniforme du vecteur tourné résultant sur une profondeur de 3 bits par coordonnée ;  
c) Le stockage dudit vecteur quantifié dans une table de pages mémoire compactée ;  
d) La validation de la préservation de distance géométrique par projection de Johnson-Lindenstrauss quantifiée comparée à un seuil d'erreur toléré avant restitution de la mémoire.

### Revendication 3 (Indépendante — Inférence Spéculative Asservie au Carbone)
Système d'inférence pour modèle de langage à décodage spéculatif, comprenant un modèle cible et un modèle brouillon, **caractérisé en ce qu'il comprend** :
- Une interface de communication réseau recevant un flux télémétrique de l'intensité d'émission carbone instantanée d'un réseau électrique régional ;
- Un contrôleur d'échantillonnage adaptant dynamiquement la longueur de séquence de spéculation $K$ du modèle brouillon en fonction inverse de l'intensité carbone reçue, augmentant le nombre $K$ de tokens spéculés lorsque l'intensité carbone diminue et réduisant le nombre $K$ lorsque l'intensité carbone augmente.

### Revendication 4 (Indépendante — Système de Calcul)
Système informatique d'accélération d'apprentissage et d'inférence de modèles de langage comprenant un processeur hôte et au moins un accélérateur tensoriel, **caractérisé en ce qu'il met en œuvre** le procédé d'attention déterministe selon la revendication 1 et le procédé de compression de cache selon la revendication 2.

### Revendications 5 à 10 (Dépendantes)
- **Revendication 5** : Procédé selon la revendication 1, dans lequel ladite table LUT est chargée en mémoire SRAM partagée d'une unité de calcul graphique (GPU) ou d'un processeur tensoriel systolique.
- **Revendication 6** : Procédé selon la revendication 2, dans lequel l'opérateur de rotation SplitMix64 garantit une conservation de la norme euclidienne à moins de 35% d'écart relatif sans perte d'expressivité de perplexité.
- **Revendication 7** : Procédé selon la revendication 3, dans lequel ladite interface est synchronisée avec l'interface de programmation applicative Eco2Mix de RTE (Réseau de Transport d'Électricité).
- **Revendication 8** : Système selon la revendication 4, comprenant en outre un ordonnanceur de tâches assignant dynamiquement les calculs d'attention aux cœurs tensoriels et les accès mémoire aux cœurs de gestion selon un ratio de défauts de cache L1 surveillé par unité de télémétrie PMU.
- **Revendication 9** : Système selon la revendication 4, dans lequel la sécurité mémoire des blocs d'allocation sans fragmentation est formellement certifiée par preuve mathématique interactive dans un environnement de vérification formelle de théorèmes (Lean 4).
- **Revendication 10** : Procédé d'apprentissage distribué selon la revendication 4, dans lequel les gradients calculés sont comprimés à un bit de signe par paramètre avec agrégation par vote majoritaire avant échange All-Reduce inter-nœuds.

---

## 7. ABRÉGÉ TECHNIQUE (ABSTRACT POUR LE BOPI)

L'invention concerne un procédé et un système d'optimisation d'inférence et d'entraînement pour réseaux de neurones transformeurs. Le système résout les problèmes de non-déterminisme numérique et d'explosion mémoire du cache clé-valeur (KV-cache). 
Une unité de calcul exécute les opérations d'attention au moyen d'un opérateur d'attention déterministe à virgule fixe entière (INT64) couplé à une table de consultation (LUT) exponentielle en SRAM, garantissant une reproductibilité bit-à-bit stricte avec zéro dérive numérique.
Conjointement, le cache clé-valeur est comprimé à 3 bits par projection orthogonale pseudo-aléatoire (PolarQuant) éliminant les valeurs aberrantes, générant un gain de mémoire d'au moins 4,92×. 
Le décodage spéculatif est asservi en temps réel à l'intensité carbone de la grille électrique pour réduire la consommation environnementale.

L'invention est destinée aux centres de calcul, aux GPU NVIDIA Hopper/Blackwell, aux TPU Google Cloud et aux processeurs vectoriels RISC-V.

---
*(c) 2026 Xavier Callens / Socrate AI Lab. Tous droits réservés.*
