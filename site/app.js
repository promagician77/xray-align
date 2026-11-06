"use strict";
const $ = (s, e = document) => e.querySelector(s), $$ = (s, e = document) => [...e.querySelectorAll(s)];
const fmt = (v, d = 2) => (v == null || isNaN(v) ? "—" : Number(v).toFixed(d));
const sgn = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const G = "#15936A", R = "#D0454B", INK = "#0F1722";
const RENDER = {}; let CUR = null;
const risk = (deg, t50 = 5.3, s = 1.0) => 1 / (1 + Math.exp(-(Math.abs(deg) - t50) / s));

function tabs() {
  const b = $$("nav.tabs button");
  const show = (id) => { b.forEach((x) => x.setAttribute("aria-selected", x.dataset.tab === id)); $$("section.tab").forEach((s) => s.classList.toggle("on", s.id === "s-" + id)); CUR = id; try { history.replaceState(null, "", "#" + id); } catch (e) {} requestAnimationFrame(() => RENDER[id] && RENDER[id]()); };
  b.forEach((x) => x.addEventListener("click", () => show(x.dataset.tab)));
  const h = location.hash.slice(1); show(b.some((x) => x.dataset.tab === h) ? h : "demo");
  let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => RENDER[CUR] && RENDER[CUR](), 150); });
}

function scatter(host, o) {
  host.innerHTML = ""; host.classList.add("chart");
  const W = Math.max(320, host.clientWidth || 600), H = o.height || 320, m = { l: 48, r: 14, t: 12, b: 36 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, host);
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r), Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const g = el("g", { class: "grid" }, svg), ax = el("g", { class: "ax" }, svg);
  ticks(y0, y1, 6).forEach((v) => { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, ax).textContent = v; });
  ticks(x0, x1, 6).forEach((v) => { el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, ax).textContent = v; });
  el("line", { x1: X(x0), y1: Y(x0), x2: X(x1), y2: Y(x1), stroke: "#9AA5B1", "stroke-dasharray": "4 4" }, svg);
  el("text", { x: W - m.r, y: H - 4, "text-anchor": "end", class: "lbl", style: "fill:#5D6977;font-size:11.5px" }, svg).textContent = o.xLabel;
  el("text", { x: m.l, y: m.t - 1, class: "lbl", style: "fill:#5D6977;font-size:11.5px" }, svg).textContent = o.yLabel;
  const tip = document.createElement("div"); tip.className = "tip"; host.appendChild(tip);
  o.series.forEach((s) => s.pts.forEach(([x, y, info]) => {
    const c = el("circle", { cx: X(x), cy: Y(Math.max(y0, Math.min(y1, y))), r: 4.5, fill: s.color, "fill-opacity": 0.75, stroke: "#fff", "stroke-width": 1.2 }, svg);
    c.addEventListener("pointerenter", () => { const r = svg.getBoundingClientRect(); tip.innerHTML = `${s.name}<br>true ${sgn(x, 2)}°, measured ${sgn(y, 2)}°<br>${info}`; tip.style.left = (X(x) / W) * r.width + "px"; tip.style.top = (Y(y) / H) * r.height + "px"; tip.classList.add("on"); });
    c.addEventListener("pointerleave", () => tip.classList.remove("on"));
  }));
}

async function main() {
  tabs();
  const j = async (p) => (await fetch(p)).json();
  const [A, F, LOG] = await Promise.all([j("data/accuracy.json"), j("data/risk_fit.json"), j("data/demo_log.json")]);
  const set = (k, v) => $$(`[data-n="${k}"]`).forEach((e) => (e.textContent = v));
  const r = A.ring, s = A.single;
  set("ring-rms", fmt(r.angle_err_rms_deg, 2)); set("single-rms", fmt(s.angle_err_rms_deg, 1)); set("ring-flips", r.sign_flips); set("flip-n", r.sign_checks);
  set("single-flips", `${s.sign_flips} of ${s.sign_checks} (${fmt(s.sign_flip_pct, 1)}%)`); set("ms", fmt(A.detect_pose_ms.mean, 0));
  set("single-mean", fmt(A.jitter.single_mean_deg, 1)); set("single-risk", Math.round(100 * risk(A.jitter.single_mean_deg)));

  $("#logtb tbody").innerHTML = LOG.map((x) => `<tr class="${x.feedback === "acceptable" ? "hl" : ""}"><td>${x.id}</td><td>${x.ts.slice(11)}</td><td class="n">${sgn(x.horizontal_deg, 2)}°</td><td class="n">${sgn(x.vertical_deg, 2)}°</td><td class="n">${fmt(x.total_deg, 2)}°</td><td class="n">${sgn(x.dx_mm, 1)}, ${sgn(x.dy_mm, 1)} mm</td><td class="n">${fmt(x.distance_mm, 0)} mm</td><td class="n">${x.markers_seen}</td><td class="n">${fmt(x.risk_pct, 1)}%</td><td><span class="pill a">${x.risk_model}</span></td><td><span class="pill ${x.feedback === "acceptable" ? "g" : "r"}">${x.feedback}</span></td></tr>`).join("");

  $("#acc-tiles").innerHTML = [
    [`${fmt(r.angle_err_rms_deg, 2)}°`, `RMS angle error, whole ring (95% of frames under ${fmt(r.angle_err_p95_deg, 2)}°, worst ${fmt(r.angle_err_max_deg, 2)}°)`],
    [`${fmt(r.near_aligned_err_rms_deg, 2)}°`, "RMS error when the tube is within 3° of aligned, where single markers are weakest"],
    [`${fmt(r.pos_err_rms_mm, 2)} mm`, `RMS centring error; distance ${fmt(r.dist_err_rms_mm, 2)} mm RMS`],
    [`${A.calibration.views} views`, `calibration: ${fmt(A.calibration.rms_px, 3)} px RMS, focal length within ${fmt(Math.abs(A.calibration.fx_err_pct), 2)}%`],
  ].map(([b, t]) => `<div class="tile"><b>${b}</b><span>${t}</span></div>`).join("");

  const pts = (k) => A.points.filter((p) => p[k]).map((p) => [p.true_h, p[k].h, `${p.d} mm, error ${fmt(p[k].err_deg, 2)}°`]);
  RENDER.accuracy = () => {
    const ds = [100, 150, 200, 250];
    lineChart($("#ch-dist"), { height: 290, xDomain: [90, 260], yDomain: [0, 9], yTicks: 6, xLabel: "tube-to-ring distance (mm)", yLabel: "RMS angle error (°)", series: [
      { name: "whole ring", color: G, x: ds, y: ds.map((d) => r.rms_by_distance_deg[d]), width: 2.5 },
      { name: "single marker", color: R, x: ds, y: ds.map((d) => s.rms_by_distance_deg[d]), width: 2.5 }], xFmtTip: (v) => `${v} mm`, yFmtTip: (v) => `${fmt(v, 2)}°` });
    scatter($("#ch-scatter"), { height: 300, x: [-16, 16], y: [-24, 24], xLabel: "true horizontal angle (°)", yLabel: "measured (°)", series: [{ name: "single marker", color: R, pts: pts("single") }, { name: "whole ring", color: G, pts: pts("ring") }] });
  };
  RENDER.ring = () => {
    const J = A.jitter, n = J.series_ring.length, xs = [...Array(n).keys()];
    lineChart($("#ch-jitter"), { height: 280, xDomain: [0, n - 1], yDomain: [-11, -5], yTicks: 6, xLabel: "frame", yLabel: "measured horizontal angle (°)", refLines: [{ y: -7, label: "true −7.0°" }], series: [
      { name: "whole ring", color: G, x: xs, y: J.series_ring, width: 2 }, { name: "single marker", color: R, x: xs, y: J.series_single, width: 1.6 }], xFmtTip: (v) => `frame ${v}`, yFmtTip: (v) => `${fmt(v, 2)}°` });
    $("#jit-note").textContent = `Standard deviation over 60 frames: ring ${fmt(J.ring_std_deg, 3)}°, single marker ${fmt(J.single_std_deg, 3)}°. Mean: ring ${fmt(J.ring_mean_deg, 2)}°, single ${fmt(J.single_mean_deg, 2)}° (a bias, not just noise).`;
    $("#ringtb tbody").innerHTML = [["RMS angle error", `${fmt(r.angle_err_rms_deg, 2)}°`, `${fmt(s.angle_err_rms_deg, 2)}°`], ["95th percentile", `${fmt(r.angle_err_p95_deg, 2)}°`, `${fmt(s.angle_err_p95_deg, 2)}°`], ["Worst frame", `${fmt(r.angle_err_max_deg, 2)}°`, `${fmt(s.angle_err_max_deg, 2)}°`], ["Direction flipped (|angle| ≥ 2°)", `${r.sign_flips} of ${r.sign_checks}`, `${s.sign_flips} of ${s.sign_checks}`], ["Distance error RMS", `${fmt(r.dist_err_rms_mm, 2)} mm`, `${fmt(s.dist_err_rms_mm, 2)} mm`]]
      .reduce((h, [a, b, c]) => h + `<tr><td>${a}</td><td class="n" style="color:${G}">${b}</td><td class="n" style="color:${R}">${c}</td></tr>`, `<tr><th></th><th class="n">Whole ring</th><th class="n">Single marker</th></tr>`);
  };
  RENDER.risk = () => {
    const xs = Array.from({ length: 121 }, (_, i) => i / 10);
    lineChart($("#ch-curve"), { height: 280, xDomain: [0, 12], yDomain: [0, 1], yTicks: 5, yFmt: (v) => Math.round(v * 100) + "%", xLabel: "total angular deviation (°)", yLabel: "estimated risk (SIMULATED)", refLines: [{ y: 0.85, label: "7° → 85% (brief's example)" }], series: [{ name: "risk", color: R, x: xs, y: xs.map((x) => risk(x)), width: 2.5 }], xFmtTip: (v) => `${fmt(v, 1)}°`, yFmtTip: (v) => `${Math.round(v * 100)}%` });
    const ns = F.by_n.map((b) => b.n);
    lineChart($("#ch-fit"), { height: 280, xDomain: [0, 520], yDomain: [2.5, 6.5], yTicks: 4, xLabel: "phantom exposures with a verdict", yLabel: "fitted 50%-risk angle (°)", refLines: [{ y: F.true_theta50, label: "value used to make the test labels" }], series: [
      { name: "5th percentile", color: G, x: ns, y: F.by_n.map((b) => b.theta50_p5), width: 2 }, { name: "95th percentile", color: G, x: ns, y: F.by_n.map((b) => b.theta50_p95), width: 2 }], xFmtTip: (v) => `n = ${v}`, yFmtTip: (v) => `${fmt(v, 2)}°` });
    const b20 = F.by_n[0], b200 = F.by_n.find((b) => b.n === 200);
    $("#fit-note").textContent = `${F.trials} simulated feedback sets per size. With 20 verdicts, the fitted 50%-risk angle lands between ${fmt(b20.theta50_p5, 1)}° and ${fmt(b20.theta50_p95, 1)}° (90% of runs). With 200 it lands between ${fmt(b200.theta50_p5, 1)}° and ${fmt(b200.theta50_p95, 1)}°. The labels come from a made-up curve; the point is how much data a stable estimate needs.`;
  };
  RENDER.camera = () => {
    const c = A.calibration;
    $("#caltb tbody").innerHTML = [["Views used", c.views], ["Reprojection error (RMS)", `${fmt(c.rms_px, 3)} px`], ["Focal length error", `${sgn(c.fx_err_pct, 3)}%`], ["Principal point error", `${sgn(c.cx_err_px, 2)}, ${sgn(c.cy_err_px, 2)} px`], ["Radial distortion k1", `${fmt(c.k1_est, 4)} (true ${fmt(c.k1_true, 4)})`]].map(([a, b]) => `<tr><td>${a}</td><td class="n">${b}</td></tr>`).join("");
    $("#zerotb tbody").innerHTML = A.zeroing.cases.map((z) => `<tr><td class="n">${sgn(z.true_h, 1)}°, 0.0°</td><td class="n" style="color:${R}">${sgn(z.no_zero_h, 2)}°, ${sgn(z.no_zero_v, 2)}°</td><td class="n" style="color:${G}">${sgn(z.zeroed_h, 2)}°, ${sgn(z.zeroed_v, 2)}°</td></tr>`).join("");
  };
  RENDER[CUR] && RENDER[CUR]();
}
main().catch((e) => { console.error(e); document.body.insertAdjacentHTML("afterbegin", `<p style="padding:10px;background:#FBE6E7">Data failed to load (${e.message}). Serve this folder over HTTP.</p>`); });
