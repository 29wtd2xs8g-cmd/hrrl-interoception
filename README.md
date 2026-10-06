# HRRL et interoception : satiation et bruit intéroceptif

Ce projet contient un agent DQN dans une gridworld à plusieurs variables internes. La récompense est homéostatique (HRRL) : c'est la réduction du drive de Keramati & Gutkin (2014). L'agent est comparé à deux contrôles, puis soumis à un bruit intéroceptif.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Lancer les expériences

```bash
# Point 4 : satiation, conditions A-D x 30 seeds, 300 000 pas d'entraînement par run
.venv/bin/python run_satiation.py --seeds 30 --workers 8
.venv/bin/python plot_satiation.py            # -> results/satiation/fig_satiation.png

# Variante temporelle : pénurie annoncée, gamma ∈ {0.5, 0.9, 0.99} + contrôle sans indice
.venv/bin/python run_scarcity.py --seeds 30 --workers 8
.venv/bin/python plot_scarcity.py             # -> results/scarcity/fig_scarcity.png
```

Chaque run est sauvegardé dans son propre fichier `.npz`, avec sa configuration complète. Si une campagne est interrompue, il suffit de relancer la même commande : les runs déjà faits sont sautés. Un `summary.csv` récapitule les métriques par seed.

Temps de calcul : environ 4 min par run quand 8 runs tournent en parallèle. La satiation compte 7 conditions × 30 seeds, soit environ 2 h sur 8 cœurs. La pénurie compte 4 × 30 runs, soit environ 1 h.

Pour un test rapide : `--seeds 2 --train-steps 30000 --out results/smoke`.

## Structure

| Fichier | Rôle |
|---|---|
| `hrrl/config.py` | Tous les paramètres (environnement, récompense, agent, run) |
| `hrrl/env.py` | Gridworld, dynamique interne, pénurie, `Interoception` (bruit) |
| `hrrl/rewards.py` | Drive de Keramati & Gutkin et les récompenses des conditions A/D, B et C |
| `hrrl/dqn.py` | Double DQN avec replay buffer |
| `hrrl/experiment.py` | Un run : entraînement, puis une vie d'évaluation de 5000 pas |
| `hrrl/metrics.py` | Métriques de satiation et de stockage |

## Modèle

- **État interne** h ∈ ℝ² (énergie, eau), consigne h* = 10, plafond h_max = 20 (capacité de l'estomac). La mort survient quand une variable atteint 0.
- **Dépense** : une perte basale de 0,04 par pas sur chaque variable, plus un coût de mouvement (0,04 en énergie, 0,02 en eau). La nourriture et l'eau sont dans des coins opposés : aller chercher l'une coûte de l'autre.
- **Drive** : D(h) = (Σᵢ |h*ᵢ − hᵢ|ⁿ)^(1/m), avec n = 4 et m = 3. Il est symétrique, donc le dépassement est pénalisé.
- **Conditions** :
  - A : HRRL, r = D(ĥₜ) − D(ĥₜ₊₁) ;
  - B : r = quantité ingérée ;
  - C : r = 1 par bouchée tant que le quota de la fenêtre glissante n'est pas atteint. Ce quota est calé sur le besoin métabolique, et la récompense est aveugle à h ;
  - D : comme A, avec ĥ = h + 𝒩(0, σ²). Le même tirage alimente l'observation **et** la récompense.
- **Observation** : position (one-hot), écart perçu (ĥ − h*)/h*, et, dans la variante pénurie, l'indice d'alerte, le drapeau de pénurie et le temps restant.

## Choix à discuter

1. **Pénalité de mort (`death_penalty = 20`)**, appliquée à toutes les conditions. Sans elle, en HRRL pure, la mort coupe les récompenses négatives, et le DQN apprenait des politiques qui se figent jusqu'à la mort dans les déficits profonds. Mettre `0` pour la HRRL pure.
2. **B et C meurent beaucoup**, même avec la pénalité. Pour le maximiseur, 20 vaut à peu près 20 bouchées : mourir peut être « rentable » par rapport à rester collé à une seule ressource. C'est un résultat en soi (une récompense non homéostatique ne protège pas l'organisme), mais c'est aussi un facteur confondant pour les métriques de satiation. On peut en tenir compte via `death_penalty` ou en rapportant les métriques par vie.
3. **Tolérance « près de la consigne » = ±2**, calculée par variable. Avec deux ressources éloignées, les variables oscillent en opposition de phase.
4. **Bruit σ** exprimé dans les unités de h (1 bouchée = 1). Les valeurs testées sont {0 ; 0,5 ; 1 ; 2 ; 4}.

## Référence

Keramati, M., & Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. *eLife*, 3, e04811.
