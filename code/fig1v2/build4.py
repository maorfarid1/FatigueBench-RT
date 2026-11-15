"""Fig. 1 v4: pipeline of numbered stage cards, orthogonal connectors with rounded corners."""
import base64, numpy as np
O = "/home/claude/paper/code/fig1v2/"
def img(f): return "data:image/png;base64," + base64.b64encode(open(O + f, "rb").read()).decode()
NAVY, BLUE, TEAL, CORAL, INK = "#1d3557", "#3d6aa8", "#2a9d8f", "#d1495b", "#1b1b1b"
W = [222, 228, 452, 188, 246, 318]; GAP = 40; X0 = 18
L = []; x = X0
for w in W: L.append(x); x += w + GAP
Rr = [l + w for l, w in zip(L, W)]; C = [l + w / 2 for l, w in zip(L, W)]
TOP, BOT, YM = 66, 482, 275
def card(i, title, n, fill="#ffffff", stroke="#d6deea", dashed=False):
    d = ' stroke-dasharray="7 6"' if dashed else ''
    s = f'<rect x="{L[i]}" y="{TOP}" width="{W[i]}" height="{BOT-TOP}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{d} filter="url(#sh)"/>'
    tw = len(title) * 10.2; bx = C[i] - (tw + 34) / 2
    s += f'<circle cx="{bx+13}" cy="{TOP+34}" r="13" fill="{NAVY}"/><text x="{bx+13}" y="{TOP+40.5}" text-anchor="middle" font-size="17" style="fill:#fff" class="b">{n}</text>'
    s += f'<text x="{bx+34}" y="{TOP+42}" font-size="22" style="fill:{NAVY}">{title}</text>'
    return s
def net(cx, cy, col, s=0.85):
    Ls = [4, 6, 6, 1]; xs = [cx - 66 * s + i * 44 * s for i in range(4)]; o = []
    pts = [[(xs[i], cy + (j - (n - 1) / 2) * 13 * s) for j in range(n)] for i, n in enumerate(Ls)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b: o.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="{col}" stroke-opacity="0.22" stroke-width="0.9"/>')
    for colp in pts:
        for p in colp: o.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="4.6" fill="url(#g{col[1:]})" stroke="{col}" stroke-width="1.1"/>')
    return "\n".join(o)
A = f'fill="none" stroke="{NAVY}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"'
def elbow(x0, y0, xt, y1, x1, arrow=True, r=14):
    sg = 1 if y1 > y0 else -1
    p = f'M{x0},{y0} H{xt-r} Q{xt},{y0} {xt},{y0+sg*r} V{y1-sg*r} Q{xt},{y1} {xt+r},{y1} H{x1}'
    return f'<path d="{p}" {A}{" marker-end=&quot;url(#ar)&quot;" if False else ""}{" marker-end=\"url(#ar)\"" if arrow else ""}/>'
# PI-BNN container internals
NL, NR = L[2] + 24, Rr[2] - 24; NY = [(128, 262), (316, 450)]; NC = [(a + b) / 2 + 14 for a, b in NY]
def ncard(y0, y1, col, grad, head, sym, t1, t2, shade):
    s = f'<rect x="{NL}" y="{y0}" width="{NR-NL}" height="{y1-y0}" rx="14" fill="#fff" stroke="{shade}" stroke-width="1.4"/>'
    s += f'<path d="M{NL+14},{y0} H{NR-14} Q{NR},{y0} {NR},{y0+14} V{y0+32} H{NL} V{y0+14} Q{NL},{y0} {NL+14},{y0} Z" fill="url(#{grad})"/>'
    s += f'<text x="{(NL+NR)/2}" y="{y0+24}" text-anchor="middle" font-size="20" style="fill:#fff">{head}</text>'
    yc = (y0 + 32 + y1) / 2
    s += net(NL + 95, yc, col)
    s += f'<text x="{NL+196}" y="{yc+8}" font-size="25" class="i">{sym}</text>'
    s += f'<text x="{NL+258}" y="{yc-4}" font-size="20" style="fill:#333">{t1}</text><text x="{NL+258}" y="{yc+20}" font-size="20" style="fill:#333">{t2}</text>'
    return s, yc
n1, y1c = ncard(*NY[0], BLUE, "hdB", "Damage-state network", "ΔD", "remaining", "damage", "#cdd8ea")
n2, y2c = ncard(*NY[1], TEAL, "hdT", "Load-forecast network", "r", "future", "damage rate", "#c7e6e0")
# clock
CX, CY, R = C[5], 252, 82
def pol(a, r): return CX + r * np.sin(a), CY - r * np.cos(a)
ticks = "\n".join(f'<line x1="{pol(a,R-4)[0]:.1f}" y1="{pol(a,R-4)[1]:.1f}" x2="{pol(a,R-(15 if i%3==0 else 9))[0]:.1f}" y2="{pol(a,R-(15 if i%3==0 else 9))[1]:.1f}" stroke="{NAVY}" stroke-width="{2.6 if i%3==0 else 1.5}" stroke-linecap="round"/>' for i, a in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)))
a1, a2, am = np.radians(62), np.radians(128), np.radians(88)
p1, p2 = pol(a1, R - 3), pol(a2, R - 3); h = pol(am, R - 14)
wedge = f'M{CX},{CY} L{p1[0]:.1f},{p1[1]:.1f} A{R-3},{R-3} 0 0,1 {p2[0]:.1f},{p2[1]:.1f} Z'
ft = pol(np.radians(45), R + 22); ci = pol(np.radians(110), R + 24)
# connectors
T1 = (Rr[1] + L[2]) / 2 + 2; T2 = (Rr[2] + L[3]) / 2
conn = [
    f'<path d="M{Rr[0]},{YM} H{L[1]-2}" {A} marker-end="url(#ar)"/>',
    elbow(Rr[1], YM, T1, y1c, NL - 2), elbow(Rr[1], YM, T1, y2c, NL - 2),
    elbow(NR, y1c, T2, YM, L[3] - 2), elbow(NR, y2c, T2, YM, L[3] - 2, arrow=False),
    f'<path d="M{Rr[3]},{YM} H{L[4]-2}" {A} marker-end="url(#ar)"/>',
    f'<path d="M{Rr[4]},{YM} H{L[5]-2}" {A} marker-end="url(#ar)"/>',
]
RY, rr = 580, 18
loop = f'<path d="M{C[5]},{BOT} V{RY-rr} Q{C[5]},{RY} {C[5]-rr},{RY} H{C[0]+rr} Q{C[0]},{RY} {C[0]},{RY-rr} V{BOT+4}" fill="none" stroke="{NAVY}" stroke-width="2.6" stroke-linecap="round" marker-end="url(#ar)"/>'
PW = 700; PX = (C[0] + C[5]) / 2 - PW / 2
pill = (f'<rect x="{PX}" y="{RY-21}" width="{PW}" height="42" rx="21" fill="#fff" stroke="{NAVY}" stroke-width="1.6"/>'
        f'<g transform="translate({PX+30},{RY})" fill="none" stroke="{CORAL}" stroke-width="2.4" stroke-linecap="round"><path d="M9,-6 A11,11 0 1,0 11,3"/><path d="M5,-8 L10,-6 L10,-12" stroke-linejoin="round"/></g>'
        f'<text x="{PX+PW/2+14}" y="{RY+8}" text-anchor="middle" font-size="22" style="fill:{NAVY}">every operating block: measure, update, predict (about 1 ms)</text>')
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{{font-family:LM;src:url("file:///usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf");}}
@font-face{{font-family:LM;font-style:italic;src:url("file:///usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf");}}
@font-face{{font-family:LM;font-weight:bold;src:url("file:///usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf");}}
html,body{{margin:0;background:#fff}} svg{{font-family:LM,serif;fill:{INK}}} .s{{font-size:20px;fill:#333}} .i{{font-style:italic}} .b{{font-weight:bold}}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="1880" height="576" viewBox="0 44 1880 576">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10L2.5 5z" fill="{NAVY}"/></marker>
<filter id="sh" x="-10%" y="-10%" width="120%" height="125%"><feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#1d3557" flood-opacity="0.10"/></filter>
<radialGradient id="g{BLUE[1:]}" cx="35%" cy="35%"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#bcd0ee"/></radialGradient>
<radialGradient id="g{TEAL[1:]}" cx="35%" cy="35%"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#b5e3dc"/></radialGradient>
<linearGradient id="hdB" x1="0" x2="1"><stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="#5b86c2"/></linearGradient>
<linearGradient id="hdT" x1="0" x2="1"><stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="#52b8aa"/></linearGradient>
<linearGradient id="pib" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f3f6fb"/><stop offset="1" stop-color="#eef6f4"/></linearGradient>
</defs>
{card(0, "Measurement", 1)}{card(1, "Fatigue analysis", 2)}{card(2, "PI-BNN: learned corrections", 3, fill="url(#pib)", stroke="#b9c6da", dashed=True)}
{card(3, "Composition", 4)}{card(4, "Posterior PDF", 5)}{card(5, "Predicted FT and CI", 6)}
<image href="{img('m_spec.png')}" x="{C[0]-57}" y="122" width="114" height="123"/>
<text class="s" x="{C[0]}" y="272" text-anchor="middle">mechanical properties</text>
<image href="{img('m_signal.png')}" x="{C[0]-108}" y="292" width="210" height="120"/>
<text class="s" x="{C[0]}" y="440" text-anchor="middle">measured signal</text>
<image href="{img('m_miner.png')}" x="{C[1]-86}" y="124" width="172" height="104"/>
<text class="s" x="{C[1]}" y="260" text-anchor="middle">rainflow, Goodman,</text><text class="s" x="{C[1]}" y="284" text-anchor="middle">Basquin, Miner</text>
<line x1="{C[1]-62}" y1="310" x2="{C[1]+62}" y2="310" stroke="#d6deea" stroke-width="1.3"/>
<text x="{C[1]}" y="372" text-anchor="middle" font-size="42" class="b" style="fill:{NAVY}">z<tspan dy="9" font-size="26" class="i" font-weight="normal">k</tspan></text>
<text class="s" x="{C[1]}" y="430" text-anchor="middle">scale-free features</text>
{n1}{n2}
<circle cx="{C[3]}" cy="245" r="44" fill="#fff" stroke="{NAVY}" stroke-width="2.8"/>
<text x="{C[3]}" y="236" text-anchor="middle" font-size="25" class="i">ΔD</text>
<line x1="{C[3]-25}" y1="246" x2="{C[3]+25}" y2="246" stroke="{INK}" stroke-width="1.6"/>
<text x="{C[3]}" y="271" text-anchor="middle" font-size="25" class="i">r</text>
<text class="s" x="{C[3]}" y="340" text-anchor="middle"><tspan class="i">RUL</tspan> = Δ<tspan class="i">D</tspan>/<tspan class="i">r</tspan></text>
<text class="s" x="{C[3]}" y="372" text-anchor="middle">+ conformal</text><text class="s" x="{C[3]}" y="396" text-anchor="middle">calibration</text>
<image href="{img('m_pdf.png')}" x="{C[4]-111}" y="126" width="222" height="178"/>
<text class="s" x="{C[4]}" y="350" text-anchor="middle">zero before the current</text><text class="s" x="{C[4]}" y="374" text-anchor="middle">time <tspan class="i">t</tspan> (Heaviside)</text>
<circle cx="{CX}" cy="{CY}" r="{R}" fill="#fff" stroke="{NAVY}" stroke-width="3"/>
{ticks}
<path d="{wedge}" fill="{CORAL}" fill-opacity="0.26" stroke="{CORAL}" stroke-width="1.4"/>
<line x1="{CX}" y1="{CY}" x2="{h[0]:.1f}" y2="{h[1]:.1f}" stroke="{CORAL}" stroke-width="4.5" stroke-linecap="round"/>
<circle cx="{CX}" cy="{CY}" r="6" fill="{NAVY}"/>
<text x="{ft[0]:.1f}" y="{ft[1]:.1f}" text-anchor="middle" font-size="22">FT</text>
<text x="{ci[0]:.1f}" y="{ci[1]+8:.1f}" text-anchor="middle" font-size="22">CI</text>
<text class="s" x="{CX}" y="388" text-anchor="middle">predicted failure time and</text><text class="s" x="{CX}" y="412" text-anchor="middle">interval at certainty level CL</text>
{"".join(conn)}
{loop}{pill}
</svg></body></html>'''
open(O + "fig1v4.html", "w").write(html)
