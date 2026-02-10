import { renderHeroStats } from "./components/hero-stats.js";
import { renderCarousel } from "./components/carousel.js";
import { renderLeaderboard } from "./components/leaderboard.js";
import { renderFindings } from "./components/findings-charts.js";
import { initReveal } from "./components/scroll-reveal.js";
import { initNav } from "./components/nav.js";

document.addEventListener("DOMContentLoaded", () => {
    renderHeroStats();
    renderCarousel();
    renderLeaderboard();
    renderFindings();
    initReveal();
    initNav();
});
