import { LEADERBOARD_DATA } from "../data/leaderboard.js";

function fmt(v) {
    return v != null ? v.toFixed(1) : "\u2014";
}

export function renderLeaderboard() {
    const tbody = document.getElementById("leaderboardBody");
    if (!tbody) return;

    // Models with scores first (sorted by masterBall desc), then pending (alphabetical)
    const scored = LEADERBOARD_DATA.filter(r => r.masterBall != null).sort((a, b) => b.masterBall - a.masterBall);
    const pending = LEADERBOARD_DATA.filter(r => r.masterBall == null).sort((a, b) => a.model.localeCompare(b.model));
    const sorted = [...scored, ...pending];

    tbody.innerHTML = sorted
        .map((row) => {
            const hasScore = row.masterBall != null;
            const rank = hasScore ? scored.indexOf(row) + 1 : "\u2014";
            const rankClass = hasScore && rank <= 3 ? ` rank-${rank}` : "";
            const scoreClass = hasScore && rank <= 3 ? ' class="score-highlight"' : "";
            return `
            <tr>
                <td class="rank-cell${rankClass}">${rank}</td>
                <td>${row.model}</td>
                <td${scoreClass}>${fmt(row.masterBall)}</td>
                <td>${fmt(row.ultraBall)}</td>
                <td>${fmt(row.greatBall)}</td>
                <td class="shadow-highlight">${fmt(row.shadowBall)}</td>
            </tr>`;
        })
        .join("");
}
