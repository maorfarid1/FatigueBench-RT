import base64, numpy as np
O = "/home/claude/paper/code/fig1v2/"
def img(f): return "data:image/png;base64," + base64.b64encode(open(O + f, "rb").read()).decode()
NAVY, BLUE, TEAL, CORAL = "#1d3557", "#3d6aa8", "#2a9d8f", "#d1495b"
def net(cx, cy, col, s=1.0):
    L = [4, 6, 6, 1]; xs = [cx - 66 * s + i * 44 * s for i in range(4)]; o = []
    pts = [[(xs[i], cy + (j - (n - 1) / 2) * 13 * s) for j in range(n)] for i, n in enumerate(L)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b: o.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="{col}" stroke-opacity="0.22" stroke-width="0.9"/>')
    for i, colp in enumerate(pts):
        for p in colp: o.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="4.6" fill="url(#g{col[1:]})" stroke="{col}" stroke-width="1.1"/>')
    return "\n".join(o)
CX, CY, R = 1690, 248, 88
def pol(a, r): return CX + r * np.sin(a), CY - r * np.cos(a)
ticks = "\n".join(f'<line x1="{pol(a, R - 4)[0]:.1f}" y1="{pol(a, R - 4)[1]:.1f}" x2="{pol(a, R - (16 if i % 3 == 0 else 10))[0]:.1f}" y2="{pol(a, R - (16 if i % 3 == 0 else 10))[1]:.1f}" stroke="{NAVY}" stroke-width="{2.6 if i % 3 == 0 else 1.6}"/>' for i, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)))
a1, a2, am = np.radians(62), np.radians(128), np.radians(88)
p1, p2 = pol(a1, R - 3), pol(a2, R - 3); h = pol(am, R - 14)
wedge = f'M{CX},{CY} L{p1[0]:.1f},{p1[1]:.1f} A{R-3},{R-3} 0 0,1 {p2[0]:.1f},{p2[1]:.1f} Z'
ft = pol(np.radians(45), R + 22); ci = pol(np.radians(108), R + 24)
svg = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{{font-family:LM;src:url("file:///usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf");}}
@font-face{{font-family:LM;font-style:italic;src:url("file:///usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf");}}
@font-face{{font-family:LM;font-weight:bold;src:url("file:///usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf");}}
html,body{{margin:0;background:#fff}} svg{{font-family:LM,serif;fill:#1b1b1b}}
.st{{font-size:29px;fill:{NAVY}}} .c{{font-size:23px}} .s{{font-size:21px;fill:#333}} .i{{font-style:italic}} .b{{font-weight:bold}}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1880" height="640" viewBox="0 0 1880 640">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10L2.5 5z" fill="{NAVY}"/></marker>
<filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#1d3557" flood-opacity="0.13"/></filter>
<radialGradient id="g{BLUE[1:]}" cx="35%" cy="35%"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#bcd0ee"/></radialGradient>
<radialGradient id="g{TEAL[1:]}" cx="35%" cy="35%"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#b5e3dc"/></radialGradient>
<linearGradient id="hdB" x1="0" x2="1"><stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="#5b86c2"/></linearGradient>
<linearGradient id="hdT" x1="0" x2="1"><stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="#52b8aa"/></linearGradient>
</defs>
<g text-anchor="middle">
<text class="st" x="140" y="42">Measurement</text><text class="st" x="405" y="42">Fatigue analysis</text>
<text class="st" x="755" y="42">Learned corrections</text><text class="st" x="1062" y="42">Composition</text>
<text class="st" x="1370" y="42">Posterior PDF</text><text class="st" x="1690" y="42">Pred. FT and CI</text>
</g>
<!-- measurement -->
<image href="{img('m_spec.png')}" x="68" y="70" width="140" height="151"/>
<text class="c" x="140" y="250" text-anchor="middle">Mechanical properties</text>
<image href="{img('m_signal.png')}" x="25" y="272" width="235" height="134"/>
<text class="c" x="140" y="430" text-anchor="middle">Measured signal</text>
<path d="M266,92 C284,92 284,104 284,124 L284,232 C284,248 290,252 300,252 C290,252 284,256 284,272 L284,380 C284,400 284,412 266,412" fill="none" stroke="{NAVY}" stroke-width="2.4"/>
<line x1="301" y1="252" x2="318" y2="252" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- fatigue analysis card -->
<rect x="322" y="96" width="168" height="314" rx="14" fill="#fbf8f2" stroke="#d8cdb4" stroke-width="1.4" filter="url(#sh)"/>
<image href="{img('m_miner.png')}" x="328" y="112" width="158" height="96"/>
<text class="s" x="406" y="236" text-anchor="middle">rainflow, Goodman,</text>
<text class="s" x="406" y="260" text-anchor="middle">Basquin, Miner</text>
<line x1="350" y1="282" x2="462" y2="282" stroke="#d8cdb4" stroke-width="1.2"/>
<text x="406" y="336" text-anchor="middle" font-size="42" class="b" style="fill:{NAVY}">z<tspan dy="9" font-size="26" class="i" font-weight="normal">k</tspan></text>
<text class="s" x="406" y="386" text-anchor="middle">scale-free features</text>
<path d="M494,200 C522,200 526,172 548,170" fill="none" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<path d="M494,306 C522,306 526,334 548,336" fill="none" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- network cards -->
<g filter="url(#sh)">
<rect x="566" y="104" width="374" height="128" rx="14" fill="#f4f7fc" stroke="#cdd8ea"/>
<rect x="558" y="96" width="374" height="128" rx="14" fill="#fff" stroke="#cdd8ea" stroke-width="1.4"/>
<rect x="566" y="270" width="374" height="128" rx="14" fill="#f2faf8" stroke="#c7e6e0"/>
<rect x="558" y="262" width="374" height="128" rx="14" fill="#fff" stroke="#c7e6e0" stroke-width="1.4"/>
</g>
<path d="M572,96 L918,96 Q932,96 932,110 L932,128 L558,128 L558,110 Q558,96 572,96 Z" fill="url(#hdB)"/>
<text x="745" y="120" text-anchor="middle" font-size="22" style="fill:#fff">Damage-state network</text>
<path d="M572,262 L918,262 Q932,262 932,276 L932,294 L558,294 L558,276 Q558,262 572,262 Z" fill="url(#hdT)"/>
<text x="745" y="286" text-anchor="middle" font-size="22" style="fill:#fff">Load-forecast network</text>
{net(650, 176, BLUE, 0.85)}
{net(650, 342, TEAL, 0.85)}
<text class="c i" x="742" y="170">ΔD</text><text class="s" x="790" y="170">remaining</text><text class="s" x="790" y="194">damage</text>
<text class="c i" x="746" y="336">r</text><text class="s" x="790" y="336">future</text><text class="s" x="790" y="360">damage rate</text>
<path d="M562,418 Q562,432 578,432 L728,432 Q744,432 747,448 Q750,432 766,432 L916,432 Q932,432 932,418" fill="none" stroke="{NAVY}" stroke-width="2"/>
<text class="c" x="747" y="480" text-anchor="middle">PI-BNN: physics prior + learned correction,</text>
<text class="c" x="747" y="508" text-anchor="middle">deep ensemble of <tspan class="i">M</tspan> = 5</text>
<!-- composition -->
<path d="M936,166 C978,170 990,206 1014,226" fill="none" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<path d="M936,330 C978,326 990,292 1014,272" fill="none" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<circle cx="1062" cy="250" r="46" fill="#fff" stroke="{NAVY}" stroke-width="3" filter="url(#sh)"/>
<text x="1062" y="241" text-anchor="middle" font-size="26" class="i">ΔD</text>
<line x1="1036" y1="251" x2="1088" y2="251" stroke="#1b1b1b" stroke-width="1.6"/>
<text x="1062" y="276" text-anchor="middle" font-size="26" class="i">r</text>
<text class="s" x="1062" y="330" text-anchor="middle"><tspan class="i">RUL</tspan> = Δ<tspan class="i">D</tspan>/<tspan class="i">r</tspan></text>
<text class="s" x="1062" y="356" text-anchor="middle">+ conformal</text><text class="s" x="1062" y="380" text-anchor="middle">calibration</text>
<line x1="1110" y1="250" x2="1248" y2="250" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- posterior pdf -->
<image href="{img('m_pdf.png')}" x="1258" y="118" width="222" height="178"/>
<text class="s" x="1370" y="330" text-anchor="middle">zero before the current</text>
<text class="s" x="1370" y="354" text-anchor="middle">time <tspan class="i">t</tspan> (Heaviside)</text>
<line x1="1494" y1="250" x2="1584" y2="250" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#ar)"/>
<!-- clock -->
<circle cx="{CX}" cy="{CY}" r="{R}" fill="#fff" stroke="{NAVY}" stroke-width="3.2" filter="url(#sh)"/>
{ticks}
<path d="{wedge}" fill="{CORAL}" fill-opacity="0.28" stroke="{CORAL}" stroke-width="1.4"/>
<line x1="{CX}" y1="{CY}" x2="{h[0]:.1f}" y2="{h[1]:.1f}" stroke="{CORAL}" stroke-width="4.5" stroke-linecap="round"/>
<circle cx="{CX}" cy="{CY}" r="6" fill="{NAVY}"/>
<text class="c" x="{ft[0]:.1f}" y="{ft[1]:.1f}" text-anchor="middle">FT</text>
<text class="c" x="{ci[0]:.1f}" y="{ci[1]+8:.1f}" text-anchor="middle">CI</text>
<text class="s" x="{CX}" y="370" text-anchor="middle">predicted failure time and</text>
<text class="s" x="{CX}" y="394" text-anchor="middle">interval at certainty level CL</text>
<!-- feedback -->
<path d="M{CX},408 L{CX},560 Q{CX},578 {CX-18},578 L158,578 Q140,578 140,560 L140,438" fill="none" stroke="{NAVY}" stroke-width="2.6" marker-end="url(#ar)"/>
<rect x="760" y="560" width="660" height="36" rx="18" fill="#fff"/>
<text class="c" x="1090" y="586" text-anchor="middle" style="fill:{NAVY}">every operating block: measure, update, predict (about 1 ms)</text>
</svg></body></html>'''
open(O + "fig1v3.html", "w").write(svg)
