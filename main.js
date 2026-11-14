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
  const HANDLE_R = 10;                 // handle dot radius (css px)
  const MARK_W = 1.5;                  // stroke of the handle line and pin cross
  const SNAP = Math.PI / 12;
  // canvas colours follow the CSS theme tokens (read again whenever the theme changes)
  const COL = { line: "#e9c454", accent: "#ffd23f", shade: "rgba(10,9,6,0.6)", bgRGB: [0.04, 0.035, 0.024] };
  function readColours() {
    const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
    COL.line = v("--line"); COL.accent = v("--accent"); COL.shade = v("--shade");
    const bg = v("--bg"); COL.bgRGB = [1, 3, 5].map((i) => parseInt(bg.slice(i, i + 2), 16) / 255);
  }

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
  let turns = 0, comparing = false;
  // view: where the image sits on the stage (css px). s = css px per image px; fit = the s
  // that shows the whole image; zoom = s / fit (1 = whole image)
  let view = { s: 1, x: 0, y: 0, w: 0, h: 0, sw: 1, sh: 1, fit: 1 };
  let zoom = 1;
  const PAD = 16, MAX_PX = 8;          // margin around the image; max zoom: 8 css px per image px

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
  uniform int uMode, uActive, uClip;
  uniform vec4 uMap;                 // image uv = vUv * uMap.zw + uMap.xy (zoom and pan)
  uniform vec3 uBg;                  // page background, shown around the image
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

  // mirror fold into n segments over span: a triangle wave with sharp creases at the mirror
  // lines; soft edges rounds those creases toward a cosine wave (same range, no corners).
  // When n isn't whole, the piece left over where the circle closes (behind the handle)
  // becomes a smaller mirror segment of its own, so it closes without a seam and grows
  // into a full segment as n rises.
  float foldN(float src, float n, float span) {
    float seg = span / n;
    float x = src + span * 0.5;
    float k = floor(x / seg), m = x - k * seg;
    float w = seg;
    if (k * seg < span && (k + 1.0) * seg > span) w = max(span - k * seg, 1e-4);
    float tri = w * 0.5 - abs(m - w * 0.5);
    float smoothFold = w * 0.25 * (1.0 - cos(TAU * min(m, w) / w));
    return mix(max(tri, -seg), smoothFold, uSoft) - seg * 0.25;
  }

  // the picture at one turn angle th, for a pixel at distance d and wedge-relative angle rel;
  // w is how much of the effect reaches this pixel (wedge feather × nothing else)
  vec4 ringAt(float th, float d, float rel, float w, bool partial) {
    float src = wrapPi(rel - th);
    if (uFold > 1.0) {                                      // kaleidoscope
      // below 2× the fold uses 2 segments while it eases in; above 2× the count keeps rising
      float span = partial ? uWedge : TAU;
      float folded = foldN(src, max(uFold, 2.0), span);
      // fades with the wedge feather and the reach like the turn itself, and fades in over
      // the first 45° of turning, so an unturned line changes nothing
      // bend from the plain direction toward the folded one. Blending the two angles would
      // flip to the other way round at one angle and leave a seam there; blending them as
      // directions (vectors) bends smoothly everywhere
      float f = foldAmount(d, w);
      if (f > 0.0) {
        vec2 dir = mix(vec2(cos(src), sin(src)), vec2(cos(folded), sin(folded)), f);
        src = atan(dir.y, dir.x);
      }
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
    vec2 uv = vUv * uMap.zw + uMap.xy;
    gX = dFdx(uv); gY = dFdy(uv);
    if (uClip == 1 && (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0)) { o = vec4(uBg, 1.0); return; }
    vec2 p = uv * uSize;
    if (uActive == 0) { o = vec4(textureGrad(uBase, uv, gX, gY).rgb, 1.0); return; }
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

  let texOrig = null, texBase = null, undoStack = [], redoStack = [];

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
    // the screen shows the stage, with the image placed by the view (zoom, pan); anything
    // rendered into a texture is the whole image at its own size
    if (target) { gl.uniform4f(U.uMap, 0, 0, 1, 1); gl.uniform1i(U.uClip, 0); }
    else {
      gl.uniform4f(U.uMap, -view.x / view.w, -view.y / view.h, view.sw / view.w, view.sh / view.h);
      gl.uniform1i(U.uClip, 1); gl.uniform3f(U.uBg, ...COL.bgRGB);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function renderScreen() {
    if (!W) return;
    // compare: show the original while held, with no marks
    if (comparing) { drawTo(null, glc.width, glc.height, false, texOrig, false); octx.clearRect(0, 0, ov.width, ov.height); return; }
    // a pin whose line is still being drawn has no length or direction yet: show the image as is
    drawTo(null, glc.width, glc.height, false, texBase, !!pivot && drag !== "line");
    drawOverlay();
  }

  function fit() {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    ov.width = Math.round(r.width * dpr); ov.height = Math.round(r.height * dpr);
    if (!W) return;
    // keep whatever is in the middle of the stage in the middle when the stage changes size
    const c = view.w ? { x: (view.sw / 2 - view.x) / view.s, y: (view.sh / 2 - view.y) / view.s } : null;
    view.sw = r.width; view.sh = r.height;
    view.fit = Math.min((r.width - 2 * PAD) / W, (r.height - 2 * PAD) / H);
    view.s = view.fit * zoom; view.w = W * view.s; view.h = H * view.s;
    if (c) { view.x = r.width / 2 - c.x * view.s; view.y = r.height / 2 - c.y * view.s; }
    clampView();
    Object.assign(glc.style, { left: "0px", top: "0px", width: r.width + "px", height: r.height + "px" });
    glc.width = Math.round(r.width * dpr); glc.height = Math.round(r.height * dpr);
  }
  // an image smaller than the stage sits in the middle; a bigger one can be moved until
  // its edge reaches the margin, never further
  function clampView() {
    const lim = (pos, size, room) => size <= room - 2 * PAD ? Math.round((room - size) / 2)
      : Math.min(PAD, Math.max(room - PAD - size, pos));
    view.x = lim(view.x, view.w, view.sw); view.y = lim(view.y, view.h, view.sh);
  }
  // zoom to z, keeping the image point under stage point (sx, sy) where it is
  function zoomTo(z, sx = view.sw / 2, sy = view.sh / 2, anchor) {
    if (!W) return;
    const a = anchor || { x: (sx - view.x) / view.s, y: (sy - view.y) / view.s };
    zoom = Math.max(1, Math.min(Math.max(1, MAX_PX / view.fit), z));
    if (Math.abs(zoom - 1) < 0.01) zoom = 1;
    view.s = view.fit * zoom; view.w = W * view.s; view.h = H * view.s;
    view.x = sx - a.x * view.s; view.y = sy - a.y * view.s;
    clampView(); schedule();
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
    // the parts you grab (handle line, pin cross, dot): in the accent colour, no outline,
    // a touch heavier than the 1px guides so they read over a busy photo
    const strokePath = (path) => { octx.strokeStyle = COL.line; octx.lineWidth = MARK_W; octx.beginPath(); path(); octx.stroke(); };
    // the reach circle: a thin 1px guide, no outline
    const R = P.reach * refR * view.s;
    const guide = (path) => { octx.strokeStyle = COL.line; octx.lineWidth = 1; octx.beginPath(); path(); octx.stroke(); };
    if (isFinite(P.reach) && drag !== "line") guide(() => octx.arc(px, py, R, 0, Math.PI * 2));
    strokePath(() => {                                              // line and pin cross
      octx.moveTo(px, py); octx.lineTo(hx, hy);
      octx.moveTo(px - 8, py - 8); octx.lineTo(px + 8, py + 8);
      octx.moveTo(px + 8, py - 8); octx.lineTo(px - 8, py + 8);
    });
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
  // history: every image state is a texture. Undo moves the current state onto the redo
  // stack; any new change (a locked-in turn, dice, reset, a new image) clears redo.
  const inUse = (t) => t === texOrig || t === texBase || undoStack.some((u) => u.tex === t) || redoStack.some((u) => u.tex === t);
  function drop(list) { for (const u of list.splice(0)) if (!inUse(u.tex)) gl.deleteTexture(u.tex); }
  function pushUndo() {
    drop(redoStack);
    undoStack.push({ tex: texBase, turns });
    if (undoStack.length > MAX_UNDO) { const u = undoStack.shift(); if (!inUse(u.tex)) gl.deleteTexture(u.tex); }
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
  // If you switch away and come straight back without turning in the other mode, the
  // lock-in is taken back and the turn is live again, so its sliders still act on it.
  let parked = null;
  function setMode(m) {
    if (m === P.mode) return;
    const pk = parked; parked = null;
    if (pk && pk.mode === m && pivot === pk.pivot && angle === 0 && texBase === pk.after &&
        undoStack.length && undoStack[undoStack.length - 1].tex === pk.before) {
      undoStack.pop(); texBase = pk.before; turns = pk.turns;
      if (!inUse(pk.after)) gl.deleteTexture(pk.after);
      refR = pk.refR; phi0 = pk.phi0; angle = pk.angle; free = 0;
      P.mode = m; afterChange(); return;
    }
    // the handle stays exactly where it is: its current direction becomes the new 0°
    const keep = pivot ? { pivot, refR, phi0: phi0 + angle } : null;
    const live = pivot && angle !== 0 ? { mode: P.mode, pivot, refR, phi0, angle, before: texBase, turns } : null;
    commit();
    if (live) parked = { ...live, after: texBase };
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
    if (pivot && angle !== 0) commit();      // lock the live turn in first, so redo can bring it back
    pivot = null; angle = 0; diceLive = false;
    const prev = undoStack.pop();
    if (prev) {
      redoStack.push({ tex: texBase, turns });
      if (redoStack.length > MAX_UNDO) { const u = redoStack.shift(); if (!inUse(u.tex)) gl.deleteTexture(u.tex); }
      texBase = prev.tex; turns = prev.turns;
    }
    schedule();
  }

  function redo() {
    const next = redoStack.pop();
    if (!next) return;
    pivot = null; angle = 0; diceLive = false;   // a turn still being made is dropped
    undoStack.push({ tex: texBase, turns });
    texBase = next.tex; turns = next.turns;
    schedule();
  }

  function reset() {
    pivot = null; angle = 0;
    const old = texBase;
    texBase = texOrig; turns = 0;
    drop(undoStack); drop(redoStack);
    if (!inUse(old)) gl.deleteTexture(old);
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
    { id: "redo-btn", text: "redo", run: redo },
    { id: "reset-btn", text: "reset", run: reset },
    { id: "dice-btn", text: "dice", run: dice },
    { id: "compare-btn", text: "compare", hold: true },
    { id: "save-btn", text: "save", run: savePNG },
  ];

  // JetBrains Mono's ∞ outline (font units: 1000 per em, baseline at 0, y up)
  // cropped tight to the glyph, then sized up ~1.5× (in CSS) so it reads as big as the digits
  const INF_SVG = '<svg class="inf" viewBox="35 -505 530 345" aria-label="infinity" role="img">' +
    '<path transform="scale(1,-1)" fill="currentColor" d="M166 160Q107 160 71.0 197.0Q35 234 35 295V370Q35 431 73.0 468.0Q111 505 172 505Q274 505 310 406H313Q318 428 332.5 450.5Q347 473 372.5 489.0Q398 505 434 505Q493 505 529.0 468.0Q565 431 565 370V295Q565 234 527.5 197.0Q490 160 428 160Q326 160 290 259H287Q283 238 268.0 215.0Q253 192 228.0 176.0Q203 160 166 160ZM175 220Q204 220 222.0 231.5Q240 243 250.5 259.0Q261 275 266.0 289.0Q271 303 272 309L241 393Q219 445 175 445Q144 445 124.5 424.5Q105 404 105 370V295Q105 261 124.5 240.5Q144 220 175 220ZM425 220Q457 220 476.0 240.5Q495 261 495 295V370Q495 404 476.0 424.5Q457 445 425 445Q397 445 378.5 433.5Q360 422 349.5 406.0Q339 390 334.0 376.0Q329 362 328 356L359 272Q381 220 425 220Z"/></svg>';
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
    if (a.hold) {        // compare: press and hold to see the original, let go to come back
      b.addEventListener("pointerdown", (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); setCompare(true); });
      for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, () => setCompare(false));
      b.addEventListener("contextmenu", (e) => e.preventDefault());
    } else b.addEventListener("click", () => { if (W || a.id === "open-btn") a.run(); });
    $("actions").appendChild(b); actionEls[a.id] = b;
  }
  $("actions").insertBefore($("theme-btn"), actionEls["save-btn"]);   // … compare, light/dark, save

  function setCompare(v) {
    if (comparing === v || !W) return;
    comparing = v; actionEls["compare-btn"].setAttribute("aria-pressed", String(v)); schedule();
  }

  // dark / light background; remembered on this device
  const themeBtn = $("theme-btn");
  function setTheme(t) {
    document.documentElement.dataset.theme = t;
    themeBtn.textContent = t === "dark" ? "light" : "dark";        // the button names where it goes
    document.querySelector('meta[name="theme-color"]').content = t === "dark" ? "#0a0906" : "#f6f4ee";
    try { localStorage.setItem("pivot.theme", t); } catch (e) {}
    readColours(); schedule();
  }
  themeBtn.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
  let savedTheme = "dark";
  try { savedTheme = localStorage.getItem("pivot.theme") || "dark"; } catch (e) {}
  setTheme(savedTheme);

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
      const txt = s.fmt(P[s.key]), val = c.querySelector(".val");
      // Google Fonts' JetBrains Mono leaves out ∞, so browsers draw it in a fallback font at
      // the wrong size. Draw JetBrains Mono's own ∞ instead, in the text colour.
      if (txt === "∞") val.innerHTML = INF_SVG; else val.textContent = txt;
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
    // a new image starts a fresh history: free every texture from the old one
    const old = new Set([texOrig, texBase, ...undoStack.map((u) => u.tex), ...redoStack.map((u) => u.tex)]);
    for (const t of old) if (t) gl.deleteTexture(t);
    gl.activeTexture(gl.TEXTURE0);
    texOrig = makeTex(W, H, c); texBase = texOrig; undoStack = []; redoStack = [];
    turns = 0; pivot = null; angle = 0;
    zoom = 1; view.w = 0;                  // a new image starts showing whole
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

  // pointers on the stage: one draws and turns; Space + drag (or the middle button) moves
  // the view; two fingers pinch to zoom and move it
  const pointers = new Map();
  let spaceDown = false, panFrom = null, pinch = null, before = null;
  const stageXY = (e) => { const r = stage.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

  ov.addEventListener("pointerdown", (e) => {
    if (!W || (e.button !== 0 && e.button !== 1)) return;
    ov.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, stageXY(e));
    if (pointers.size === 2 && e.pointerType === "touch") { startPinch(); return; }
    if (pointers.size > 1 || drag) return;
    if (spaceDown || e.button === 1) {
      e.preventDefault();
      drag = "pan"; panFrom = { ...stageXY(e), x0: view.x, y0: view.y };
      document.body.classList.add("panning");
    } else if (nearHandle(e.clientX, e.clientY)) {
      drag = "turn";
      const p = toImg(e.clientX, e.clientY);
      lastRaw = Math.atan2(p.y - pivot.y, p.x - pivot.x);
      free = angle;
      document.body.classList.add("grabbing");
    } else {
      // remember the state, so a second finger arriving (a pinch, not a new line) can put it back
      before = { texBase, turns, pivot, angle, phi0, refR, diceLive };
      commit();
      pivot = toImg(e.clientX, e.clientY); handle = { ...pivot };
      drag = "line";
    }
    schedule();
  });

  function startPinch() {
    if (drag === "line" && before) {                // the first finger wasn't drawing after all
      if (texBase !== before.texBase) {
        const t = texBase; undoStack.pop(); texBase = before.texBase;
        if (!inUse(t)) gl.deleteTexture(t);
      }
      ({ turns, pivot, angle, phi0, refR, diceLive } = before);
      handle = null;
    }
    document.body.classList.remove("grabbing");
    const [a, b] = [...pointers.values()];
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, z: zoom,
              anchor: { x: (mid.x - view.x) / view.s, y: (mid.y - view.y) / view.s } };
    drag = "pinch"; schedule();
  }

  ov.addEventListener("pointermove", (e) => {
    if (!W) return;
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, stageXY(e));
    if (drag === "pinch") {
      if (pointers.size < 2) return;
      const [a, b] = [...pointers.values()];
      zoomTo(pinch.z * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d, (a.x + b.x) / 2, (a.y + b.y) / 2, pinch.anchor);
      return;
    }
    if (drag === "pan") {
      const p = stageXY(e);
      view.x = panFrom.x0 + p.x - panFrom.x; view.y = panFrom.y0 + p.y - panFrom.y;
      clampView(); schedule(); return;
    }
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
    const was = drag; drag = null; before = null;
    document.body.classList.remove("grabbing", "panning");
    if (was === "line") {
      const [px, py] = toScr(pivot), [hx, hy] = toScr(handle);
      if (Math.hypot(hx - px, hy - py) < GRAB_R * 1.5) pivot = null; // a click, not a line
      else placeLine(pivot, handle);
      handle = null;
    }
    schedule();
  }
  function pointerUp(e) {
    pointers.delete(e.pointerId);
    if (drag === "pinch") { if (pointers.size === 0) drag = null; return; }   // lift both fingers to finish
    endDrag();
  }
  ov.addEventListener("pointerup", pointerUp);
  ov.addEventListener("pointercancel", pointerUp);
  ov.addEventListener("auxclick", (e) => e.preventDefault());
  // scroll wheel / trackpad pinch: zoom toward the pointer
  ov.addEventListener("wheel", (e) => {
    if (!W) return;
    e.preventDefault();
    const dy = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1);
    const p = stageXY(e);
    zoomTo(zoom * Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.002)), p.x, p.y);
  }, { passive: false });

  // ---- keys -----------------------------------------------------------------
  const cycle = (arr, v) => arr[(arr.indexOf(v) + 1) % arr.length];
  addEventListener("keyup", (e) => {
    if (e.key.toLowerCase() === "c") setCompare(false);
    if (e.key === " ") { spaceDown = false; document.body.classList.remove("pan"); }
  });
  addEventListener("blur", () => { setCompare(false); spaceDown = false; document.body.classList.remove("pan"); });
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape" && !info.hidden) { closeInfo(); return; }
    if (!info.hidden || !W) return;
    if (e.key === " " && !(e.target instanceof HTMLInputElement)) {   // hold Space to move the view
      e.preventDefault(); spaceDown = true; document.body.classList.add("pan"); return;
    }
    if (drag) return;
    if (e.target instanceof HTMLInputElement && e.key.startsWith("Arrow")) return;
    switch (e.key.toLowerCase()) {
      case "m": setMode(cycle(MODES, P.mode)); break;
      case "arrowleft": case "arrowright":
        if (!pivot) return;
            angle += (e.key === "ArrowRight" ? 1 : -1) * (Math.PI / 180) * (e.shiftKey ? 15 : 1);
        afterChange(); break;
      case "d": dice(); break;
      case "z": e.shiftKey ? redo() : undo(); break;
      case "y": redo(); break;
      case "r": reset(); break;
      case "s": savePNG(); break;
      case "c": setCompare(true); break;
      case "l": themeBtn.click(); break;
      case "0": zoomTo(1); break;
      case "+": case "=": zoomTo(zoom * 1.25); break;
      case "-": case "_": zoomTo(zoom / 1.25); break;
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
