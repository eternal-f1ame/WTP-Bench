import { SHOWCASE_SHADOW, SHOWCASE_HQ } from "../data/showcase.js";

function statusIcon(status) {
    if (status === "correct")   return "\u2713";
    if (status === "incorrect") return "\u2717";
    return "~";
}

function buildCard(d) {
    const silhouetteClass = d.silhouette ? " silhouette-mode" : "";
    return `
    <div class="showcase-card${silhouetteClass}">
        <div class="showcase-card-image" style="background: ${d.bgGradient};">
            <span class="tier-badge tier-${d.tier}">${d.tierLabel}</span>
            <img src="${d.image}" class="pokemon-sprite" alt="Pokemon">
        </div>
        <div class="showcase-card-body">
            <h3>${d.name}</h3>
            <div class="pokemon-gen">${d.gen}</div>
            <div class="model-responses">
                ${d.responses.map(r => `
                    <div class="model-response">
                        <span class="model-name">${r.model}</span>
                        <span class="model-answer ${r.status}">${statusIcon(r.status)} ${r.answer}</span>
                    </div>
                `).join("")}
            </div>
        </div>
    </div>`;
}

function fillTrack(trackId, data) {
    const track = document.getElementById(trackId);
    if (!track) return;
    const html = data.map(buildCard).join("");
    // Duplicate for seamless infinite scroll
    track.innerHTML = html + html;
}

export function renderCarousel() {
    fillTrack("carouselTrackShadow", SHOWCASE_SHADOW);
    fillTrack("carouselTrackHQ", SHOWCASE_HQ);
}
