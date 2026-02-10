import { HERO_STATS } from "../data/stats.js";

export function renderHeroStats() {
    const container = document.getElementById("heroStats");
    if (!container) return;
    container.innerHTML = HERO_STATS.map(s => `
        <div class="stat-item">
            <div class="stat-value">${s.value}</div>
            <div class="stat-label">${s.label}</div>
        </div>
    `).join("");
}
