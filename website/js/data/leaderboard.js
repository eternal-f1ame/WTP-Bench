// Scores are cumulative accuracy (%) on HQ official artwork.
// masterBall = exact match (species + form + variant)
// ultraBall  = masterBall + correct species (wrong form/variant)
// greatBall  = ultraBall  + correct evolutionary line
// shadowBall = exact match on silhouette only
export const LEADERBOARD_DATA = [
    // Proprietary
    { model: "GPT-4.1",              masterBall: 78.97, ultraBall: 84.22, greatBall: 84.66, shadowBall: 49.48 },
    { model: "Gemini 2.5 Pro",       masterBall: 78.19, ultraBall: 88.10, greatBall: 91.38, shadowBall: 23.02 },
    { model: "Gemini 2.5 Flash",     masterBall: 77.24, ultraBall: 83.02, greatBall: 86.21, shadowBall: 30.00 },
    { model: "Claude 4.5 Sonnet",    masterBall: 65.09, ultraBall: 79.22, greatBall: 81.90, shadowBall: 53.10 },
    { model: "GPT-4o mini",          masterBall: 49.31, ultraBall: 61.98, greatBall: 68.71, shadowBall:  8.53 },
    { model: "Grok 4",               masterBall: 46.90, ultraBall: 59.05, greatBall: 63.71, shadowBall:  7.93 },
    { model: "Claude Haiku 4.5",     masterBall: 27.59, ultraBall: 35.78, greatBall: 42.07, shadowBall: 27.41 },
    // Open-weight
    { model: "Qwen3-VL 4B",          masterBall: 13.36, ultraBall: 18.45, greatBall: 22.41, shadowBall:  3.10 },
    { model: "Qwen3-VL 8B",          masterBall: 12.76, ultraBall: 17.16, greatBall: 19.74, shadowBall:  null },
    { model: "Pixtral 12B",          masterBall:  7.16, ultraBall:  9.74, greatBall: 13.28, shadowBall:  1.12 },
    { model: "Qwen2-VL 7B",          masterBall:  6.47, ultraBall:  9.22, greatBall: 12.67, shadowBall:  1.29 },
    { model: "Ovis2.5 9B",           masterBall:  5.95, ultraBall:  9.31, greatBall: 13.88, shadowBall:  2.16 },
    { model: "InternVL2.5 26B",      masterBall:  4.14, ultraBall:  6.55, greatBall: 10.17, shadowBall:  1.72 },
    { model: "LLaVA-OneVision 7B",   masterBall:  3.53, ultraBall:  6.38, greatBall:  9.91, shadowBall:  1.81 },
    { model: "Qwen2.5-VL 7B",        masterBall:  3.53, ultraBall:  5.78, greatBall:  7.16, shadowBall:  0.78 },
    { model: "Florence-VL 8B",       masterBall:  2.93, ultraBall:  5.86, greatBall:  8.45, shadowBall:  0.78 },
    { model: "Qwen2.5-VL 3B",        masterBall:  2.84, ultraBall:  5.00, greatBall:  6.55, shadowBall:  0.60 },
    { model: "Ovis2.5 2B",           masterBall:  2.76, ultraBall:  5.26, greatBall:  7.67, shadowBall:  1.12 },
    { model: "SmolVLM",              masterBall:  2.24, ultraBall:  2.24, greatBall:  2.24, shadowBall:  0.95 },
    { model: "Qwen2-VL 2B",          masterBall:  2.24, ultraBall:  4.57, greatBall:  6.03, shadowBall:  0.78 },
    { model: "PaliGemma 3B-448",     masterBall:  1.98, ultraBall:  4.48, greatBall:  5.60, shadowBall:  0.34 },
    { model: "PaliGemma 3B",         masterBall:  1.81, ultraBall:  4.14, greatBall:  5.26, shadowBall:  0.43 },
    { model: "LLaVA 1.5 13B",        masterBall:  1.81, ultraBall:  4.74, greatBall:  6.90, shadowBall:  0.69 },
    { model: "LLaVA 1.5 7B",         masterBall:  1.03, ultraBall:  3.10, greatBall:  5.09, shadowBall:  0.52 },
    { model: "InternVL2 8B",         masterBall:  0.86, ultraBall:  2.76, greatBall:  3.97, shadowBall:  0.34 },
    { model: "Phi-3 Vision 128K",    masterBall:  0.17, ultraBall:  1.03, greatBall:  1.64, shadowBall:  0.00 },
    { model: "LLaVA-OneVision 0.5B", masterBall:  null, ultraBall:  null, greatBall:  null, shadowBall:  0.09 },
    { model: "InternVL2 1B",         masterBall:  0.09, ultraBall:  0.17, greatBall:  0.26, shadowBall:  0.26 },
];
