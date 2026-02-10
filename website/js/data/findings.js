// Per-generation accuracy for proprietary vs open-weight (HQ mode)
export const GEN_ACCURACY = {
    generations: ["Gen 1", "Gen 2", "Gen 3", "Gen 4", "Gen 5", "Gen 6", "Gen 7", "Gen 8"],
    datasetCounts: [227, 111, 165, 130, 185, 109, 104, 129],
    proprietary: [64.5, 73.4, 64.7, 65.7, 62.2, 48.4, 60.6, 39.2],
    openWeight:  [ 8.6,  5.8,  2.2,  4.2,  1.8,  3.4,  1.8,  0.9],
};

// Per-type accuracy (HQ, proprietary models aggregated)
export const TYPE_ACCURACY = {
    proprietary: [
        { type: "fire",     acc: 65.71 },
        { type: "normal",   acc: 65.20 },
        { type: "poison",   acc: 64.29 },
        { type: "water",    acc: 63.62 },
        { type: "rock",     acc: 63.29 },
        { type: "fighting", acc: 63.21 },
        { type: "dark",     acc: 62.78 },
        { type: "steel",    acc: 62.61 },
        { type: "grass",    acc: 62.24 },
        { type: "fairy",    acc: 62.05 },
        { type: "ground",   acc: 60.79 },
        { type: "ice",      acc: 60.29 },
        { type: "flying",   acc: 56.86 },
        { type: "ghost",    acc: 54.20 },
        { type: "psychic",  acc: 53.46 },
        { type: "bug",      acc: 52.96 },
        { type: "dragon",   acc: 51.50 },
        { type: "electric", acc: 51.04 },
    ],
    openWeight: [
        { type: "fire",     acc: 7.00 },
        { type: "normal",   acc: 4.56 },
        { type: "flying",   acc: 4.53 },
        { type: "water",    acc: 4.36 },
        { type: "grass",    acc: 4.26 },
        { type: "fairy",    acc: 4.18 },
        { type: "electric", acc: 3.82 },
        { type: "ice",      acc: 3.60 },
        { type: "dark",     acc: 3.45 },
        { type: "psychic",  acc: 3.43 },
        { type: "poison",   acc: 3.21 },
        { type: "ground",   acc: 2.94 },
        { type: "steel",    acc: 2.72 },
        { type: "dragon",   acc: 2.43 },
        { type: "rock",     acc: 2.22 },
        { type: "ghost",    acc: 2.06 },
        { type: "fighting", acc: 1.99 },
        { type: "bug",      acc: 1.94 },
    ],
};

// Base form vs variant accuracy (HQ, top models)
export const FORM_ACCURACY = [
    { model: "GPT-4.1",          base: 90.6, mega: 87.5, gmax: 31.2, regional: 63.6, other:  7.5 },
    { model: "Gemini 2.5 Pro",   base: 92.2, mega: 52.1, gmax: 31.2, regional: 58.2, other:  5.0 },
    { model: "Gemini 2.5 Flash", base: 90.7, mega: 62.5, gmax: 31.2, regional: 49.1, other:  6.7 },
    { model: "Claude 4.5 Sonnet",base: 82.8, mega:  2.1, gmax:  0.0, regional:  9.1, other:  0.0 },
    { model: "GPT-4o mini",      base: 63.0, mega:  2.1, gmax:  0.0, regional:  1.8, other:  0.0 },
    { model: "Grok 4",           base: 59.9, mega:  2.1, gmax:  0.0, regional:  1.8, other:  0.0 },
    { model: "Claude Haiku 4.5", base: 35.2, mega:  0.0, gmax:  0.0, regional:  1.8, other:  0.0 },
];

// Gen 1 prediction bias — when wrong, what % of guesses are Gen 1?
export const GEN1_BIAS = {
    labels: ["Proprietary (HQ)", "Proprietary (Sil)", "Open-weight (HQ)", "Open-weight (Sil)"],
    gen1PctOfWrongs: [24.1, 26.5, 57.1, 79.2],
    gen1PctInDataset: 19.6,  // baseline
};

// Most confused pairs (HQ, across all models)
export const TOP_CONFUSIONS = [
    { label: "pikachu-hoenn-cap",    predicted: "pikachu",   count: 24, note: "form confusion" },
    { label: "pikachu-original-cap", predicted: "pikachu",   count: 24, note: "form confusion" },
    { label: "pikachu-partner-cap",  predicted: "pikachu",   count: 24, note: "form confusion" },
    { label: "pikachu-starter",      predicted: "pikachu",   count: 24, note: "form confusion" },
    { label: "pikachu-unova-cap",    predicted: "pikachu",   count: 24, note: "form confusion" },
    { label: "pikachu-world-cap",    predicted: "pikachu",   count: 24, note: "form confusion" },
    { label: "charizard-mega-y",     predicted: "charizard", count: 22, note: "form confusion" },
    { label: "eevee-starter",        predicted: "eevee",     count: 22, note: "form confusion" },
    { label: "vivillon",             predicted: "butterfree", count: 16, note: "visual similarity" },
];

// Universally hard — 0 correct across all 27 models on HQ
export const UNIVERSALLY_HARD = {
    hqCount: 162,     // Pokemon with 0 correct on HQ
    silCount: 327,    // Pokemon with 0 correct on silhouette
    examples: [
        "pikachu-cosplay", "pikachu-alola-cap", "deoxys-normal-attack",
        "castform-rainy", "giratina-altered-origin", "shaymin-land-sky",
        "basculin-red-striped-blue-striped", "wormadam-plant-sandy",
    ],
};

// HQ vs silhouette biggest drops (proprietary models)
export const SILHOUETTE_DROPS = [
    { pokemon: "magmortar",   hq: 100.0, sil:   0.0, drop: 100.0, type: "fire" },
    { pokemon: "slugma",      hq: 100.0, sil:  14.3, drop:  85.7, type: "fire" },
    { pokemon: "drifblim",    hq: 100.0, sil:  14.3, drop:  85.7, type: "ghost, flying" },
    { pokemon: "galvantula",  hq: 100.0, sil:  14.3, drop:  85.7, type: "bug, electric" },
    { pokemon: "alcremie",    hq: 100.0, sil:  14.3, drop:  85.7, type: "fairy" },
    { pokemon: "abomasnow",   hq:  85.7, sil:   0.0, drop:  85.7, type: "grass, ice" },
];

// Surprising: Pokemon recognized better as silhouette than HQ
export const SILHOUETTE_GAINS = [
    { pokemon: "absol",            hq:  85.7, sil: 100.0, gain: 14.3, type: "dark" },
    { pokemon: "scizor",           hq:  57.1, sil:  71.4, gain: 14.3, type: "bug, steel" },
    { pokemon: "venusaur-mega",    hq:  28.6, sil:  42.9, gain: 14.3, type: "grass, poison" },
    { pokemon: "charizard-mega-x", hq:  28.6, sil:  42.9, gain: 14.3, type: "fire, flying" },
    { pokemon: "cramorant",        hq:  42.9, sil:  57.1, gain: 14.3, type: "flying, water" },
    { pokemon: "zamazenta",        hq:  14.3, sil:  28.6, gain: 14.3, type: "fighting" },
];
