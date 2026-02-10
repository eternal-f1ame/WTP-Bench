import { GEN_ACCURACY, TYPE_ACCURACY, FORM_ACCURACY, GEN1_BIAS } from "../data/findings.js";

function heatColor(value, max = 100) {
    const ratio = value / max;
    if (ratio >= 0.7) return "rgba(34, 197, 94, 0.25)";
    if (ratio >= 0.4) return "rgba(250, 204, 21, 0.2)";
    if (ratio >= 0.1) return "rgba(255, 59, 59, 0.15)";
    return "rgba(255, 59, 59, 0.06)";
}

export function renderFindings() {
    renderGenAccuracyChart();
    renderGen1BiasChart();
    renderFormAccuracyChart();
    renderTypeAccuracyChart();
}

function renderGenAccuracyChart() {
    const el = document.getElementById("genAccuracyChart");
    if (!el) return;

    const d = GEN_ACCURACY;
    const maxVal = 80;

    let html = `<div class="grouped-bar-chart">`;
    for (let i = 0; i < d.generations.length; i++) {
        const pH = (d.proprietary[i] / maxVal) * 100;
        const oH = (d.openWeight[i] / maxVal) * 100;
        html += `
        <div class="bar-group">
            <div class="bar-group-bars">
                <div class="bar-vertical bar-fill-proprietary" style="height:${pH}%" title="${d.generations[i]}: Proprietary ${d.proprietary[i]}%"></div>
                <div class="bar-vertical bar-fill-openweight" style="height:${oH}%" title="${d.generations[i]}: Open-weight ${d.openWeight[i]}%"></div>
            </div>
            <div class="bar-group-label">${d.generations[i].replace("Gen ", "G")}</div>
        </div>`;
    }
    html += `</div>`;
    html += `<div class="chart-legend">
        <div class="chart-legend-item"><div class="chart-legend-dot" style="background:var(--accent-red)"></div> Proprietary avg</div>
        <div class="chart-legend-item"><div class="chart-legend-dot" style="background:var(--accent-blue)"></div> Open-weight avg</div>
    </div>`;
    el.innerHTML = html;
}

function renderGen1BiasChart() {
    const el = document.getElementById("gen1BiasChart");
    if (!el) return;

    const d = GEN1_BIAS;
    const maxVal = 85;

    let html = `<div class="bar-chart">`;
    for (let i = 0; i < d.labels.length; i++) {
        const w = (d.gen1PctOfWrongs[i] / maxVal) * 100;
        const isHigh = d.gen1PctOfWrongs[i] > d.gen1PctInDataset * 1.5;
        const cls = isHigh ? "bar-fill-highlight" : "bar-fill-proprietary";
        html += `
        <div class="bar-row">
            <div class="bar-label">${d.labels[i].replace("Proprietary ", "Prop ").replace("Open-weight ", "OW ")}</div>
            <div class="bar-track">
                <div class="bar-fill ${cls}" style="width:${w}%"></div>
            </div>
            <div class="bar-value">${d.gen1PctOfWrongs[i]}%</div>
        </div>`;
    }
    // Baseline
    const bw = (d.gen1PctInDataset / maxVal) * 100;
    html += `
    <div class="bar-row">
        <div class="bar-label">Dataset</div>
        <div class="bar-track">
            <div class="bar-fill bar-fill-baseline" style="width:${bw}%"></div>
        </div>
        <div class="bar-value">${d.gen1PctInDataset}%</div>
    </div>`;
    html += `</div>`;
    el.innerHTML = html;
}

function renderFormAccuracyChart() {
    const el = document.getElementById("formAccuracyChart");
    if (!el) return;

    const categories = ["Base", "Mega", "Gmax", "Regional", "Other"];
    const keys = ["base", "mega", "gmax", "regional", "other"];

    let html = `<div class="form-chart-grid">`;
    // Header
    html += `<div class="form-chart-header"></div>`;
    for (const cat of categories) {
        html += `<div class="form-chart-header">${cat}</div>`;
    }
    // Rows
    for (const row of FORM_ACCURACY) {
        html += `<div class="form-chart-model">${row.model}</div>`;
        for (const key of keys) {
            const val = row[key];
            const bg = heatColor(val);
            const color = val > 50 ? "var(--accent-green)" : val > 10 ? "var(--accent-yellow)" : "var(--text-muted)";
            html += `<div class="form-chart-cell" style="background:${bg};color:${color}">${val.toFixed(1)}%</div>`;
        }
    }
    html += `</div>`;
    el.innerHTML = html;
}

function renderTypeAccuracyChart() {
    const el = document.getElementById("typeAccuracyChart");
    if (!el) return;

    const types = TYPE_ACCURACY.proprietary;
    const maxVal = 70;

    let html = `<div class="bar-chart">`;
    for (const t of types) {
        const w = (t.acc / maxVal) * 100;
        const isTop = t.acc >= 64;
        const isBottom = t.acc <= 53;
        const cls = isTop ? "bar-fill-proprietary" : isBottom ? "bar-fill-openweight" : "bar-fill-proprietary";
        const opacity = isBottom ? "opacity:0.6" : "";
        html += `
        <div class="bar-row">
            <div class="bar-label" style="min-width:65px;text-transform:capitalize">${t.type}</div>
            <div class="bar-track">
                <div class="bar-fill ${cls}" style="width:${w}%;${opacity}"></div>
            </div>
            <div class="bar-value">${t.acc.toFixed(1)}%</div>
        </div>`;
    }
    html += `</div>`;
    html += `<div class="chart-legend">
        <div class="chart-legend-item"><div class="chart-legend-dot" style="background:var(--accent-red)"></div> Proprietary avg accuracy (HQ)</div>
    </div>`;
    el.innerHTML = html;
}
