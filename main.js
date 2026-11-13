// Pivot — pin a point, swing a handle, and the image turns around it like a wheel.
// Rendered on the GPU: every screen pixel looks up where it came from in the image
// and samples it smoothly, so turns stay clean at full screen resolution.
//   rings:   bands around the pin, each turning by its own amount
//   turn:    the whole image rotates
(() => {
  "use strict";

  // ---- settings -------------------------------------------------------------
  const MAX_SIDE = 2560;     // image long side kept on the GPU
  const MAX_UNDO = 12;
  const GRAB_R = 22;         // css px: press this close to the handle to turn it
  const HANDLE_R = 8;
  const SNAP = Math.PI / 12;
  const COL = { fg: "#f4e7b6", line: "#e9c454", accent: "#ffd23f", shade: "rgba(10,9,6,0.6)", guide: "rgba(233,196,84,0.55)" };

  const MODES = ["rings", "turn"];

  // ---- state ----------------------------------------------------------------
  const P = {
    mode: "rings", gears: 0,
    slice: 15, soft: 0, wedge: 360, taper: 0, fold: 1,
    reach: 2.5,
  };
  let W = 0, H = 0;                    // image size on the GPU
  let source = null;
  let pivot = null, handle = null;     // image px; handle only while drawing the line
  let angle = 0, refR = 1, phi0 = 0;
  let drag = null, free = 0, lastRaw = 0;
  let turns = 0;
  let view = { s: 1, x: 0, y: 0, w: 0, h: 0 };

  // ---- dom ------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const stage = $("stage"), glc = $("gl"), ov = $("overlay"), octx = ov.getContext("2d");
  const panel = $("panel"), emptyBtn = $("empty"), fileInput = $("file");
  const info = $("info"), infoBtn = $("info-btn");

  // ---- webgl ----------------------------------------------------------------
  const gl = glc.getContext("webgl2", { premultipliedAlpha: false, antialias: false });
  if (!gl) { emptyBtn.textContent = "this needs a browser with WebGL2"; emptyBtn.disabled = true; return; }

  const VS = `#version 300 es
  in vec2 aPos; out vec2 vUv; uniform int uFlip;
  void main() {
    vUv = aPos * 0.5 + 0.5;
    vUv.y = 1.0 - vUv.y;                       // uv (0,0) = image top-left
    gl_Position = vec4(aPos.x, uFlip == 1 ? -aPos.y : aPos.y, 0.0, 1.0);
  }`;

  const FS = `#version 300 es
  precision highp float; precision highp int;
  in vec2 vUv; out vec4 o;
  uniform sampler2D uBase;
  uniform vec2 uSize, uPivot;
  uniform float uAngle, uRefR, uSlice, uPhi0, uWedge, uTaper, uSoft, uFold, uReach, uGears;
  uniform int uMode, uActive;
  const float PI = 3.14159265358979, TAU = 6.28318530717959;

  float wrapPi(float a) { return a - TAU * floor((a + PI) / TAU); }
  vec2 rot(vec2 v, float a) { float c = cos(a), s = sin(a); return vec2(c*v.x - s*v.y, s*v.x + c*v.y); }

  // bands further from the pin turn more, in proportion to their distance
  float prof(float r) {
    return r / uRefR;
  }

  // reach: the turn fades out between 70% and 100% of reach × handle length (0 = no limit)
  float fall(float r) {
    if (uReach <= 0.0) return 1.0;
    float L = uReach * uRefR;
    return 1.0 - smoothstep(0.7 * L, L, r);
  }

  float bandAngle(int i) {
    float k = prof((float(i) + 0.5) * uSlice) * fall((float(i) + 0.5) * uSlice);
    // gears: every other band's turn goes from the same (0) through still (0.5) to opposite (1)
    if ((i % 2) == 1) k *= 1.0 - 2.0 * uGears;
    return uAngle * k;
  }

  void polar(vec2 v, out float d, out float a) { d = length(v); a = atan(v.y, v.x); }
  vec2 unpolar(float d, float a) { return d * vec2(cos(a), sin(a)); }

  // texture-space footprint of one screen pixel, taken before any distortion, so the right
  // mip level is used when the image is drawn smaller than its own size
  vec2 gX, gY;

  vec4 fetch(vec2 p) {
    vec2 uv = p / uSize;
    uv = 1.0 - abs(mod(uv, 2.0) - 1.0);   // beyond the frame, the image is mirrored
    return textureGrad(uBase, uv, gX, gY);
  }

  // how strongly the kaleidoscope applies here: 1×–3× eases it in; it also fades with the
  // wedge feather w and the reach, and over the first 45° of turning
  float foldAmount(float d, float w) {
    if (uFold <= 1.0) return 0.0;
    return smoothstep(1.0, 3.0, uFold) * w * fall(d) * min(1.0, abs(uAngle) / (PI / 4.0));
  }

  // the picture at one turn angle th, for a pixel at distance d and wedge-relative angle rel;
  // w is how much of the effect reaches this pixel (wedge feather × nothing else)
  vec4 ringAt(float th, float d, float rel, float w, bool partial) {
    float src = wrapPi(rel - th);
    if (uFold > 1.0) {                                      // kaleidoscope
      // below 2× the fold uses 2 segments while it eases in; above 2× the count keeps rising
      float n = max(uFold, 2.0);
      float span = partial ? uWedge : TAU;
      float seg = span / n;
      float m = mod(src + span * 0.5, seg);
      // mirror fold: a triangle wave with sharp creases at the mirror lines; soft edges
      // rounds those creases toward a cosine wave (same range, no corners)
      float tri = seg * 0.5 - abs(m - seg * 0.5);
      float smoothFold = seg * 0.25 * (1.0 - cos(TAU * m / seg));
      float folded = mix(tri, smoothFold, uSoft) - seg * 0.25;
      // fades with the wedge feather and the reach like the turn itself, and fades in over
      // the first 45° of turning, so an unturned line changes nothing
      // blend along the shorter way round, so a small amount is a small change
      src += foldAmount(d, w) * wrapPi(folded - src);
    }
    return fetch(uPivot + unpolar(d, src + uPhi0));
  }

  vec4 rings(vec2 p) {
    vec2 v = p - uPivot; float d, a;
    polar(v, d, a);
    float rel = wrapPi(a - uPhi0);
    bool partial = uWedge < TAU - 1e-4;
    // wedge: full effect inside, none outside. Taper widens a feather centred on the
    // wedge's edge lines, reaching both inward and outward, so 0% is a clean cut and
    // any value above it grows smoothly from there.
    float w = 1.0;
    if (partial) {
      float halfW = uWedge * 0.5, edge = halfW - abs(rel);
      float f = uTaper * max(halfW, PI / 6.0);
      w = f > 0.0 ? smoothstep(-f, f, edge) : step(0.0, edge);
      if (w <= 0.0) return fetch(p);
    }
    float pb = d / uSlice - 0.5;
    int i0 = int(floor(pb)); float t = fract(pb);
    if (pb < 0.0) { i0 = 0; t = 0.0; }
    // soft widens the seam between neighbouring bands (0 = crisp, anti-aliased;
    // 1 = a smooth ripple), so the bands, and their size, always stay visible
    float aa = max(uSoft * 0.5, fwidth(pb) * 0.75);
    float k = smoothstep(0.5 - aa, 0.5 + aa, t);
    float thA = bandAngle(i0) * w, thB = bandAngle(i0 + 1) * w;
    vec4 bent = ringAt(mix(thA, thB, k), d, rel, w, partial);
    // Bending the angle across a seam sweeps through everything between the two bands'
    // angles, which smears into streaks when the seam is only a pixel or two wide (hard
    // edges) or gets folded (kaleidoscope). There, crossfade the two bands' finished
    // pictures instead; soft edges keep the smooth bend.
    float crisp = max(1.0 - smoothstep(0.0, 0.15, uSoft), foldAmount(d, w));
    if (crisp > 0.0 && k > 0.0 && k < 1.0) {
      vec4 faded = mix(ringAt(thA, d, rel, w, partial), ringAt(thB, d, rel, w, partial), k);
      return mix(bent, faded, crisp);
    }
    return bent;
  }

  vec4 turnAll(vec2 p) {
    vec2 v = p - uPivot;
    return fetch(uPivot + rot(v, -uAngle * fall(length(v))));
  }

  void main() {
    gX = dFdx(vUv); gY = dFdy(vUv);
    vec2 p = vUv * uSize;
    if (uActive == 0) { o = vec4(texture(uBase, vUv).rgb, 1.0); return; }
    o = vec4((uMode == 1 ? turnAll(p) : rings(p)).rgb, 1.0);
  }`;

  function shader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const U = {};
  for (let i = 0, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name;
    U[name] = gl.getUniformLocation(prog, name);
  }
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.uniform1i(U.uBase, 0);

  function makeTex(w, h, src) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    // mipmapped, so drawing the image smaller than its own size stays clean, not jaggy
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (src) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src); gl.generateMipmap(gl.TEXTURE_2D); }
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    return t;
  }
  const fbo = gl.createFramebuffer();

  let texOrig = null, texBase = null, undoStack = [];

  // ---- drawing --------------------------------------------------------------
  function setUniforms(active) {
    gl.uniform2f(U.uSize, W, H);
    gl.uniform2f(U.uPivot, pivot ? pivot.x : 0, pivot ? pivot.y : 0);
    gl.uniform1f(U.uAngle, angle);
    gl.uniform1f(U.uRefR, refR);
    gl.uniform1f(U.uSlice, P.slice);
    gl.uniform1f(U.uPhi0, phi0);
    gl.uniform1f(U.uWedge, (P.wedge / 360) * Math.PI * 2);
    gl.uniform1f(U.uSoft, P.soft);
    gl.uniform1f(U.uTaper, P.taper);
    gl.uniform1f(U.uFold, P.fold);
    gl.uniform1f(U.uReach, isFinite(P.reach) ? P.reach : 0);
    gl.uniform1i(U.uMode, MODES.indexOf(P.mode));
    gl.uniform1f(U.uGears, P.gears);
    gl.uniform1i(U.uActive, active ? 1 : 0);
  }

  function drawTo(target, w, h, flip, tex, active) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? fbo : null);
    if (target) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(U.uFlip, flip ? 1 : 0);
    setUniforms(active);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function renderScreen() {
    if (!W) return;
    // a pin whose line is still being drawn has no length or direction yet: show the image as is
    drawTo(null, glc.width, glc.height, false, texBase, !!pivot && drag !== "line");
    drawOverlay();
  }

  function fit() {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    ov.width = Math.round(r.width * dpr); ov.height = Math.round(r.height * dpr);
    if (!W) return;
    const s = Math.min((r.width - 32) / W, (r.height - 32) / H);
    view.s = s; view.w = Math.round(W * s); view.h = Math.round(H * s);
    view.x = Math.round((r.width - view.w) / 2); view.y = Math.round((r.height - view.h) / 2);
    Object.assign(glc.style, { left: view.x + "px", top: view.y + "px", width: view.w + "px", height: view.h + "px" });
    glc.width = Math.round(view.w * dpr); glc.height = Math.round(view.h * dpr);
  }

  const toImg = (cx, cy) => {
    const r = stage.getBoundingClientRect();
    return { x: (cx - r.left - view.x) / view.s, y: (cy - r.top - view.y) / view.s };
  };
  const toScr = (p) => [view.x + p.x * view.s, view.y + p.y * view.s]; // stage css px

  function handleScr() {
    if (!pivot) return null;
    if (drag === "line") return toScr(handle);
    const [px, py] = toScr(pivot), a = phi0 + angle;
    return [px + Math.cos(a) * refR * view.s, py + Math.sin(a) * refR * view.s];
  }

  function drawOverlay() {
    const dpr = window.devicePixelRatio || 1;
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.clearRect(0, 0, ov.width, ov.height);
    if (!pivot) return;
    const [px, py] = toScr(pivot), [hx, hy] = handleScr();
    // the parts you grab (handle line, pin cross, dot): 1.5px yellow over a soft dark
    // outline, so they read on bright and dark parts of a photo alike
    const strokePath = (path) => {
      for (const [c, w] of [[COL.shade, 3.5], [COL.line, 1.5]]) {
        octx.strokeStyle = c; octx.lineWidth = w; octx.beginPath(); path(); octx.stroke();
      }
    };
    // the reach circle: a thin 1px guide, no outline
    const R = P.reach * refR * view.s;
    const guide = (path) => { octx.strokeStyle = COL.line; octx.lineWidth = 1; octx.beginPath(); path(); octx.stroke(); };
    if (isFinite(P.reach) && drag !== "line") guide(() => octx.arc(px, py, R, 0, Math.PI * 2));
    strokePath(() => {                                              // line and pin cross
      octx.moveTo(px, py); octx.lineTo(hx, hy);
      octx.moveTo(px - 7, py - 7); octx.lineTo(px + 7, py + 7);
      octx.moveTo(px + 7, py - 7); octx.lineTo(px - 7, py + 7);
    });
    octx.fillStyle = COL.shade; octx.beginPath(); octx.arc(hx, hy, HANDLE_R + 1, 0, 7); octx.fill();
    octx.fillStyle = COL.accent; octx.beginPath(); octx.arc(hx, hy, HANDLE_R, 0, 7); octx.fill();
  }

  let raf = 0;
  function schedule() { if (!raf) raf = requestAnimationFrame(frame); }

  // redraw on demand
  function frame() {
    raf = 0;
    renderScreen();
  }

  // ---- edits ----------------------------------------------------------------
  function pushUndo() {
    undoStack.push({ tex: texBase, turns });
    if (undoStack.length > MAX_UNDO) { const u = undoStack.shift(); if (u.tex !== texOrig) gl.deleteTexture(u.tex); }
  }

  function commit() {
    if (pivot && angle !== 0) {
      const next = makeTex(W, H, null);
      drawTo(next, W, H, true, texBase, true);
      pushUndo();
      gl.bindTexture(gl.TEXTURE_2D, next); gl.generateMipmap(gl.TEXTURE_2D);
      texBase = next; turns++;
    }
    pivot = handle = null; angle = 0; diceLive = false;
  }

  // switching mode locks the current turn into the image first, so nothing already done
  // changes; the pin and handle stay where they are, ready to turn in the new mode
  function setMode(m) {
    if (m === P.mode) return;
    // the handle stays exactly where it is: its current direction becomes the new 0°
    const keep = pivot ? { pivot, refR, phi0: phi0 + angle } : null;
    commit();
    if (keep) { pivot = keep.pivot; refR = keep.refR; phi0 = keep.phi0; angle = 0; free = 0; }
    P.mode = m;
    afterChange();
  }

  function placeLine(p, h) {
    pivot = p; angle = 0; free = 0;
    refR = Math.max(1, Math.hypot(h.x - p.x, h.y - p.y));
    phi0 = Math.atan2(h.y - p.y, h.x - p.x);
  }

  function setAngle(a) { angle = a; schedule(); }

  function undo() {
    if (pivot && angle !== 0) { pivot = null; angle = 0; diceLive = false; schedule(); return; }
    pivot = null; angle = 0;
    const prev = undoStack.pop();
    if (prev) { if (texBase !== texOrig && !undoStack.some((u) => u.tex === texBase)) gl.deleteTexture(texBase); texBase = prev.tex; turns = prev.turns; }
    schedule();
  }

  function reset() {
    pivot = null; angle = 0;
    for (const u of undoStack) if (u.tex !== texOrig) gl.deleteTexture(u.tex);
    if (texBase !== texOrig) gl.deleteTexture(texBase);
    undoStack = []; texBase = texOrig; turns = 0;
    schedule();
  }

  // every roll applies random effects to the original image, never on top of earlier work.
  // What you had is kept on the undo stack: undo once drops the roll, again brings it back.
  let diceLive = false;
  function dice() {
    if (!W) return;
    pivot = handle = null; angle = 0;                   // a live turn (or the last roll) is dropped
    if (texBase !== texOrig) { pushUndo(); texBase = texOrig; turns = 0; }
    const pick = (a) => a[(Math.random() * a.length) | 0];
    const rnd = (a, b) => a + Math.random() * (b - a);
    P.mode = Math.random() < 0.7 ? "rings" : pick(MODES);
    P.gears = Math.random() < 0.3 ? +rnd(0.3, 1).toFixed(2) : 0;
    P.wedge = Math.random() < 0.6 ? 360 : pick([270, 180, 120, 90]);
    P.taper = pick([0, 0.3, 0.6, 1]);
    P.slice = Math.round(pick([0.004, 0.008, 0.015, 0.03, 0.06]) * Math.max(W, H)) || 2;
    P.soft = pick([0, 0.3, 1, 1]);
    P.fold = Math.random() < 0.2 ? +rnd(2, 8).toFixed(2) : 1;
    P.reach = Math.random() < 0.2 ? Infinity : +rnd(1.5, 4.5).toFixed(2);
    const p = { x: W * rnd(0.25, 0.75), y: H * rnd(0.25, 0.75) };
    const d = Math.min(W, H) * rnd(0.12, 0.4), a0 = Math.random() * Math.PI * 2;
    placeLine(p, { x: p.x + Math.cos(a0) * d, y: p.y + Math.sin(a0) * d });
    angle = (Math.random() < 0.5 ? -1 : 1) * rnd(0.3, 1.8);
    diceLive = true;
    syncUI(); schedule();
  }

  // ---- save & record --------------------------------------------------------
  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  function savePNG() {
    if (!W) return;
    const t = makeTex(W, H, null);
    drawTo(t, W, H, true, texBase, !!pivot);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    const px = new Uint8ClampedArray(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(t);
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    c.getContext("2d").putImageData(new ImageData(px, W, H), 0, 0);
    c.toBlob((b) => b && download(b, `pivot-${turns}.png`), "image/png");
    schedule();
  }

  // ---- controls -------------------------------------------------------------
  const pct = (v) => `${Math.round(v * 100)}%`;
  const SLIDERS = [
    { key: "reach", name: "reach", min: 1, max: 8.2, step: 0.01, fmt: (v) => (isFinite(v) ? `${v.toFixed(2)}×` : "∞"),
      get: () => (isFinite(P.reach) ? P.reach : 8.2), set: (v) => (v > 8 ? Infinity : v) },
    { key: "slice", name: "ring size", min: 1, max: 240, step: 1, fmt: (v) => `${v}px`, only: ["rings"] },
    { key: "soft", name: "soft edges", min: 0, max: 1, step: 0.01, fmt: pct, only: ["rings"] },
    { key: "wedge", name: "wedge", min: 10, max: 360, step: 1, fmt: (v) => `${v}°`, only: ["rings"] },
    { key: "taper", name: "wedge taper", min: 0, max: 1, step: 0.01, fmt: pct, only: ["rings"] },
    { key: "fold", name: "kaleidoscope", min: 1, max: 12, step: 0.01, fmt: (v) => (v <= 1 ? "off" : `${v.toFixed(2)}×`), only: ["rings"] },
    { key: "gears", name: "gears", min: 0, max: 1, step: 0.01, only: ["rings"],
      fmt: (v) => (v <= 0 ? "off" : v >= 1 ? "opposite" : Math.abs(v - 0.5) < 0.005 ? "still" : pct(v)) },
  ];
  const ACTIONS = [
    { id: "open-btn", text: "open", run: () => fileInput.click() },
    { id: "undo-btn", text: "undo", run: undo },
    { id: "reset-btn", text: "reset", run: reset },
    { id: "dice-btn", text: "dice", run: dice },
    { id: "save-btn", text: "save", run: savePNG },
  ];

  const sliderEls = {}, actionEls = {};

  // mode: a two-state slider that snaps to either end. No label: the two ends are
  // named underneath (rings left, turn right) and the active one is lit.
  const modeEl = document.createElement("div");
  modeEl.className = "cell mode";
  modeEl.innerHTML = `<input id="sl-mode" type="range" min="0" max="1" step="1" aria-label="mode: rings or turn">` +
    MODES.map((m) => `<span class="end" data-mode="${m}">${m}</span>`).join("");
  const modeIn = modeEl.querySelector("input");
  modeIn.addEventListener("input", () => { if (W) setMode(MODES[+modeIn.value]); else modeIn.value = MODES.indexOf(P.mode); });
  for (const e of modeEl.querySelectorAll(".end")) e.addEventListener("click", () => { if (W) setMode(e.dataset.mode); });
  $("sliders").appendChild(modeEl);
  for (const s of SLIDERS) {
    const id = `sl-${s.key}`;
    const cell = document.createElement("label");
    cell.className = "cell"; cell.htmlFor = id;
    cell.innerHTML = `<input id="${id}" type="range" min="${s.min}" max="${s.max}" step="${s.step}"><span class="name">${s.name}</span><span class="val"></span>`;
    const input = cell.querySelector("input");
    input.addEventListener("input", () => {
      P[s.key] = s.set ? s.set(+input.value) : +input.value;
      afterChange();
    });
    $("sliders").appendChild(cell); sliderEls[s.key] = cell;
  }
  for (const a of ACTIONS) {
    const b = document.createElement("button"); b.type = "button"; b.id = a.id; b.textContent = a.text;
    b.addEventListener("click", () => { if (W || a.id === "open-btn") a.run(); });
    $("actions").appendChild(b); actionEls[a.id] = b;
  }

  function afterChange() {
    syncUI(); schedule();
  }

  function syncUI() {
    modeIn.value = MODES.indexOf(P.mode);
    for (const e of modeEl.querySelectorAll(".end")) e.classList.toggle("on", e.dataset.mode === P.mode);
    for (const s of SLIDERS) {
      const c = sliderEls[s.key];
      const offC = !!s.only && !s.only.includes(P.mode);
      c.classList.toggle("off", offC);
      c.querySelector("input").disabled = offC;
      c.querySelector("input").value = s.get ? s.get() : P[s.key];
      c.querySelector(".val").textContent = s.fmt(P[s.key]);
    }
    for (const a of ACTIONS) if (a.toggle) actionEls[a.id].setAttribute("aria-pressed", String(a.toggle()));
    requestAnimationFrame(() => { fit(); schedule(); }); // the panel may have changed height
  }

  // ---- image in -------------------------------------------------------------
  function setImage(im) {
    source = im;
    const iw = im.naturalWidth || im.width, ih = im.naturalHeight || im.height;
    const k = Math.min(1, Math.min(MAX_SIDE, gl.getParameter(gl.MAX_TEXTURE_SIZE)) / Math.max(iw, ih));
    W = Math.max(2, Math.round(iw * k)); H = Math.max(2, Math.round(ih * k));
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const g = c.getContext("2d");
    g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
    g.imageSmoothingQuality = "high";
    g.drawImage(im, 0, 0, W, H);
    for (const u of undoStack) if (u.tex !== texOrig) gl.deleteTexture(u.tex);
    if (texBase && texBase !== texOrig) gl.deleteTexture(texBase);
    if (texOrig) gl.deleteTexture(texOrig);
    gl.activeTexture(gl.TEXTURE0);
    texOrig = makeTex(W, H, c); texBase = texOrig; undoStack = [];
    turns = 0; pivot = null; angle = 0;
    document.body.classList.add("has-image");
    emptyBtn.hidden = true;
    panel.hidden = false;
    requestAnimationFrame(() => { fit(); schedule(); });
    syncUI();
  }
  function loadURL(url) { const im = new Image(); im.onload = () => setImage(im); im.src = url; }
  function loadFile(f) { if (f && f.type.startsWith("image/")) loadURL(URL.createObjectURL(f)); }

  emptyBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { loadFile(fileInput.files[0]); fileInput.value = ""; });
  addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("dragging"); });
  addEventListener("dragleave", () => document.body.classList.remove("dragging"));
  addEventListener("drop", (e) => {
    e.preventDefault(); document.body.classList.remove("dragging");
    loadFile([...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith("image/")));
  });
  addEventListener("paste", (e) => {
    const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (it) loadFile(it.getAsFile());
  });

  // ---- pointer --------------------------------------------------------------
  function nearHandle(cx, cy) {
    const h = handleScr(); if (!h) return false;
    const r = stage.getBoundingClientRect();
    return Math.hypot(cx - r.left - h[0], cy - r.top - h[1]) <= GRAB_R;
  }

  ov.addEventListener("pointerdown", (e) => {
    if (!W || e.button !== 0) return;
    ov.setPointerCapture(e.pointerId);
    if (nearHandle(e.clientX, e.clientY)) {
      drag = "turn";
      const p = toImg(e.clientX, e.clientY);
      lastRaw = Math.atan2(p.y - pivot.y, p.x - pivot.x);
      free = angle;
      document.body.classList.add("grabbing");
    } else {
      commit();
      pivot = toImg(e.clientX, e.clientY); handle = { ...pivot };
      drag = "line";
    }
    schedule();
  });

  ov.addEventListener("pointermove", (e) => {
    if (!W) return;
    if (!drag) { document.body.classList.toggle("near", nearHandle(e.clientX, e.clientY)); return; }
    const p = toImg(e.clientX, e.clientY);
    if (drag === "line") { handle = p; schedule(); return; }
    const raw = Math.atan2(p.y - pivot.y, p.x - pivot.x);
    free += Math.atan2(Math.sin(raw - lastRaw), Math.cos(raw - lastRaw)); // keep winding past 360°
    lastRaw = raw;
    setAngle(e.shiftKey ? Math.round(free / SNAP) * SNAP : free);
  });

  function endDrag() {
    if (!drag) return;
    const was = drag; drag = null;
    document.body.classList.remove("grabbing");
    if (was === "line") {
      const [px, py] = toScr(pivot), [hx, hy] = toScr(handle);
      if (Math.hypot(hx - px, hy - py) < GRAB_R * 1.5) pivot = null; // a click, not a line
      else placeLine(pivot, handle);
      handle = null;
    }
    schedule();
  }
  ov.addEventListener("pointerup", endDrag);
  ov.addEventListener("pointercancel", endDrag);
  ov.addEventListener("wheel", (e) => {
    if (!W || P.mode === "turn") return;
    e.preventDefault();
    P.slice = Math.max(1, Math.min(240, P.slice + (e.deltaY > 0 ? -1 : 1) * Math.max(1, Math.round(P.slice * 0.1))));
    afterChange();
  }, { passive: false });

  // ---- keys -----------------------------------------------------------------
  const cycle = (arr, v) => arr[(arr.indexOf(v) + 1) % arr.length];
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape" && !info.hidden) { closeInfo(); return; }
    if (!info.hidden || !W || drag) return;
    if (e.target instanceof HTMLInputElement && e.key.startsWith("Arrow")) return;
    switch (e.key.toLowerCase()) {
      case "m": setMode(cycle(MODES, P.mode)); break;
      case "arrowleft": case "arrowright":
        if (!pivot) return;
            angle += (e.key === "ArrowRight" ? 1 : -1) * (Math.PI / 180) * (e.shiftKey ? 15 : 1);
        afterChange(); break;
      case "d": dice(); break;
      case "z": undo(); break;
      case "r": reset(); break;
      case "s": savePNG(); break;
      default: return;
    }
    e.preventDefault();
  });
  addEventListener("resize", () => { fit(); schedule(); });

  // ---- info -----------------------------------------------------------------
  // info: over the image area on wide screens (landscape), from the top down to just
  // above the button on phones; the button itself stays lit and becomes "close"
  const sheet = info.querySelector(".sheet");
  const M = 16;                                     // the page's margin everywhere
  function placeInfo() {
    if (info.hidden) return;
    const b = infoBtn.getBoundingClientRect();
    if (innerWidth <= 560) {
      Object.assign(sheet.style, { left: M + "px", right: M + "px", top: `calc(${M}px + env(safe-area-inset-top, 0px))`,
        // stop one button-gap above the close button, so the edge lines up with the button grid
        bottom: (innerHeight - b.top + 8) + "px", width: "", maxHeight: "" });
      fillHatch();
    } else {
      const st = stage.getBoundingClientRect();
      const w = Math.min(st.width - 2 * M, 760);
      Object.assign(sheet.style, { width: w + "px", left: (st.left + (st.width - w) / 2) + "px", right: "",
        bottom: "", maxHeight: (st.height - 2 * M) + "px" });
      fillHatch();                                   // after sizing, so the count fits the final width
      sheet.style.top = (st.top + Math.max(M, (st.height - sheet.offsetHeight) / 2)) + "px";
    }
  }
  // fill the slash band with exactly as many characters as fit its width
  // the slashes are a filler: as many per row as fit the width, and on phones as many rows
  // as it takes to push the credit down to the bottom of the frame
  const hatch = info.querySelector(".hatch");
  const infoFrame = info.querySelector(".frame");
  const credit = info.querySelector(".credit");
  function fillHatch() {
    const probe = document.createElement("span"); probe.textContent = "/".repeat(50);
    hatch.style.letterSpacing = "0px";
    hatch.textContent = ""; hatch.appendChild(probe);
    const cw = probe.getBoundingClientRect().width / 50;
    const W = hatch.clientWidth;
    const n = Math.max(2, Math.floor(W / cw));
    // spread the leftover width evenly between the slashes, so each row runs exactly from
    // the left text edge to the right one (the spacing after the last slash is clipped)
    hatch.style.letterSpacing = Math.max(0, (W - n * cw) / (n - 1)) + "px";
    const line = "/".repeat(n);
    const set = (rows) => { hatch.textContent = Array(rows).fill(line).join("\n"); };
    if (innerWidth > 560) { set(4); return; }       // wide screens: the box fits its content
    // phones: add rows until the credit sits at the bottom of the frame
    set(2);
    infoFrame.scrollTop = 0;
    const rowH = hatch.getBoundingClientRect().height / 2;
    const fr = infoFrame.getBoundingClientRect(), cs = getComputedStyle(infoFrame);
    const inner = fr.bottom - parseFloat(cs.borderBottomWidth) - parseFloat(cs.paddingBottom);
    const spare = inner - credit.getBoundingClientRect().bottom;
    if (spare >= rowH) set(2 + Math.floor(spare / rowH));
    else if (spare < 0 && spare > -rowH) set(1);     // just short of fitting: one row makes it fit
  }

  function openInfo() { info.hidden = false; infoBtn.textContent = "close"; infoBtn.setAttribute("aria-expanded", "true"); placeInfo(); }
  function closeInfo() { info.hidden = true; infoBtn.textContent = "info"; infoBtn.setAttribute("aria-expanded", "false"); infoBtn.focus(); }
  infoBtn.setAttribute("aria-expanded", "false");
  infoBtn.addEventListener("click", () => (info.hidden ? openInfo() : closeInfo()));
  info.addEventListener("click", (e) => { if (!e.target.closest(".sheet")) closeInfo(); });
  addEventListener("resize", placeInfo);

  fit();
  if (window.PIVOT_SAMPLE) loadURL(window.PIVOT_SAMPLE);
})();
