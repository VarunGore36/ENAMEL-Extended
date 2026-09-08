(function () {
    "use strict";

    const DATA_URL = "data/results.json";

    const COLORS = {
        accent: "#2563eb",
        accentLight: "#93c5fd",
        green: "#16a34a",
        red: "#dc2626",
        muted: "#94a3b8",
        border: "#e0e0e0",
        fg: "#1a1a1a",
    };

    const MODEL_ORDER = [
        "GPT-4", "GPT-4 Turbo", "ChatGPT",
        "Phind Code Llama V2", "Code Llama 34B Python", "Code Llama 13B Python",
        "Code Llama 7B Python", "CodeGen 16B", "CodeGen 6B", "CodeGen 2B",
        "CodeT5+ 16B", "Mistral 7B", "Vicuna 13B", "Vicuna 7B",
        "SantaCoder", "Incoder 6B", "Incoder 1B",
        "GPT-J", "GPT-Neo 2B", "PolyCoder", "StableLM 7B",
        "reference", "humaneval-canonical",
    ];

    function shortDate(iso) {
        if (!iso) return "—";
        const d = new Date(iso);
        return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    }

    function shortTime(iso) {
        if (!iso) return "—";
        const d = new Date(iso);
        return d.toLocaleString("en-US", {
            month: "short", day: "numeric",
            hour: "2-digit", minute: "2-digit",
        });
    }

    function pct(v, d) {
        if (d === undefined) d = 1;
        return (v * 100).toFixed(d) + "%";
    }

    function el(tag, attrs, children) {
        const e = document.createElement(tag);
        if (attrs) Object.entries(attrs).forEach(([k, v]) => {
            if (k === "class") e.className = v;
            else if (k === "text") e.textContent = v;
            else e.setAttribute(k, v);
        });
        if (children) children.forEach(c => {
            if (typeof c === "string") e.appendChild(document.createTextNode(c));
            else if (c) e.appendChild(c);
        });
        return e;
    }

    function tdNum(v) { return el("td", { class: "num", text: v }); }
    function tdText(v) { return el("td", { text: v }); }

    // ── Charts ──────────────────────────────────────────────────────────

    function renderLeaderboardChart(canvas, leaderboard) {
        const labels = leaderboard.map(r => r.model);
        const effs = leaderboard.map(r => r.eff1);
        const passes = leaderboard.map(r => r.pass1);

        new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    { label: "eff@1", data: effs, backgroundColor: COLORS.accent, borderRadius: 3 },
                    { label: "pass@1", data: passes, backgroundColor: COLORS.muted, borderRadius: 3 },
                ],
            },
            options: {
                responsive: true,
                indexAxis: "y",
                scales: {
                    x: { max: 1, ticks: { callback: v => pct(v, 0) }, grid: { color: COLORS.border } },
                    y: { grid: { display: false } },
                },
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                    tooltip: { callbacks: { label: ctx => ctx.dataset.label + ": " + pct(ctx.parsed.x) } },
                },
            },
        });
    }

    function renderEffPassChart(canvas, leaderboard) {
        new Chart(canvas, {
            type: "scatter",
            data: {
                datasets: [{
                    data: leaderboard.map(r => ({ x: r.pass1, y: r.eff1, label: r.model })),
                    backgroundColor: COLORS.accent,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                }],
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: "pass@1" }, min: 0, max: 1, grid: { color: COLORS.border } },
                    y: { title: { display: true, text: "eff@1" }, min: 0, max: 1, grid: { color: COLORS.border } },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const d = ctx.raw;
                                return d.label + " — eff@1=" + pct(d.y) + ", pass@1=" + pct(d.x);
                            },
                        },
                    },
                },
            },
        });
    }

    function renderQDistChart(canvas, qDist) {
        const labels = qDist.map(q => "Level " + q.level);
        new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    { label: "q median", data: qDist.map(q => q.q_median), backgroundColor: COLORS.accent, borderRadius: 3 },
                    { label: "q min", data: qDist.map(q => q.q_min), backgroundColor: COLORS.accentLight, borderRadius: 3 },
                ],
            },
            options: {
                responsive: true,
                scales: {
                    y: { min: 0, max: 1.05, grid: { color: COLORS.border } },
                    x: { grid: { display: false } },
                },
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                },
            },
        });
    }

    function renderSensitivityChart(canvas, qDist) {
        const labels = qDist.map(q => "Level " + q.level);
        const scores2 = qDist.map(q => q.score_at_2x);
        const scores5 = qDist.map(q => q.score_at_5x);
        const scores10 = qDist.map(q => q.score_at_10x);

        new Chart(canvas, {
            type: "bar",
            data: {
                labels,
                datasets: [
                    { label: "Score at 2× slowdown", data: scores2, backgroundColor: "#2563eb", borderRadius: 3 },
                    { label: "Score at 5× slowdown", data: scores5, backgroundColor: "#7c3aed", borderRadius: 3 },
                    { label: "Score at 10× slowdown", data: scores10, backgroundColor: "#c026d3", borderRadius: 3 },
                ],
            },
            options: {
                responsive: true,
                scales: {
                    y: { min: 0, max: 1.05, ticks: { callback: v => pct(v, 0) }, grid: { color: COLORS.border } },
                    x: { grid: { display: false } },
                },
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                },
            },
        });
    }

    function renderHistoryChart(canvas, history) {
        const sorted = [...history].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
        const labels = sorted.map(r => shortDate(r.timestamp));
        const models = new Set();
        sorted.forEach(r => r.leaderboard.forEach(l => models.add(l.model)));

        const palette = [
            "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed",
            "#0891b2", "#c026d3", "#ea580c",
        ];
        const datasets = [...models].map((model, i) => ({
            label: model,
            data: sorted.map(r => {
                const row = r.leaderboard.find(l => l.model === model);
                return row ? row.eff1 : null;
            }),
            borderColor: palette[i % palette.length],
            backgroundColor: "transparent",
            tension: 0.3,
            pointRadius: 4,
            borderWidth: 2,
        }));

        new Chart(canvas, {
            type: "line",
            data: { labels, datasets },
            options: {
                responsive: true,
                scales: {
                    y: { min: 0, max: 1, ticks: { callback: v => pct(v, 0) }, grid: { color: COLORS.border } },
                    x: { grid: { display: false } },
                },
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                },
            },
        });
    }

    // ── Sections ────────────────────────────────────────────────────────

    function renderMeta(container, latest) {
        const cards = [
            { label: "Experiment", value: latest.experiment_id.slice(0, 19) },
            { label: "Date", value: shortDate(latest.timestamp) },
            { label: "Commit", value: latest.git_commit },
            { label: "Models", value: latest.summary.models },
            { label: "Problems", value: latest.summary.problems_scored },
            { label: "Best eff@1", value: pct(latest.leaderboard[0]?.eff1 || 0) },
        ];
        cards.forEach(c => {
            container.appendChild(
                el("div", { class: "meta-card" }, [
                    el("div", { class: "label", text: c.label }),
                    el("div", { class: "value", text: String(c.value) }),
                ])
            );
        });
    }

    function renderLeaderboardTable(tbody, leaderboard) {
        leaderboard.forEach((r, i) => {
            const tr = el("tr", null, [
                tdNum(i + 1),
                tdText(r.model),
                tdNum(pct(r.eff1)),
                tdNum(pct(r.pass1)),
                tdNum(r.problems),
                tdNum(r.correct),
                tdNum(r.censored),
            ]);
            tbody.appendChild(tr);
        });
    }

    function renderQTable(tbody, qDist) {
        qDist.forEach(q => {
            const tr = el("tr", null, [
                tdNum(q.level),
                tdNum(q.q_median.toFixed(4)),
                tdText(q.q_min.toFixed(4) + " – " + q.q_max.toFixed(4)),
                tdNum(q.tolerated_slowdown.toFixed(1) + "×"),
                tdNum(pct(q.score_at_2x)),
                tdNum(pct(q.score_at_5x)),
                tdNum(pct(q.score_at_10x)),
            ]);
            tbody.appendChild(tr);
        });
    }

    function renderExperimentsTable(tbody, history) {
        history.forEach(r => {
            const best = r.leaderboard[0]?.eff1 || 0;
            const tr = el("tr", null, [
                tdText(shortTime(r.timestamp)),
                el("td", null, [el("code", { text: r.git_commit })]),
                tdNum(r.summary.models),
                tdNum(r.summary.problems_scored),
                tdNum(pct(best)),
                tdText(r.platform?.split("-").slice(0, 2).join(" ") || "—"),
            ]);
            tbody.appendChild(tr);
        });
    }

    // ── Main ────────────────────────────────────────────────────────────

    async function init() {
        let data;
        try {
            const resp = await fetch(DATA_URL);
            if (!resp.ok) throw new Error(resp.status);
            data = await resp.json();
        } catch (e) {
            document.getElementById("results-loading").textContent =
                "Could not load results data: " + e.message;
            return;
        }

        const latest = data.latest;
        const history = data.history;

        // Results section
        document.getElementById("results-loading").style.display = "none";
        document.getElementById("results-content").style.display = "";

        renderMeta(document.getElementById("result-meta"), latest);
        renderLeaderboardChart(document.getElementById("chart-leaderboard"), latest.leaderboard);
        renderEffPassChart(document.getElementById("chart-eff-pass"), latest.leaderboard);
        renderLeaderboardTable(
            document.querySelector("#leaderboard-table tbody"),
            latest.leaderboard
        );
        renderQDistChart(document.getElementById("chart-q-dist"), latest.q_distribution);
        renderSensitivityChart(document.getElementById("chart-sensitivity"), latest.q_distribution);
        renderQTable(
            document.querySelector("#q-table tbody"),
            latest.q_distribution
        );

        // Experiments section
        document.getElementById("experiments-loading").style.display = "none";
        document.getElementById("experiments-content").style.display = "";

        renderExperimentsTable(
            document.querySelector("#experiments-table tbody"),
            history
        );
        renderHistoryChart(document.getElementById("chart-history"), history);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
