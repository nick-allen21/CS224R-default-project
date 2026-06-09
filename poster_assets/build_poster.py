"""Build the CS224R poster: 36" x 24" LANDSCAPE, tiles to 8 A4 (4x2).

Content source: poster_assets/COWORK_PROMPT.md (FINAL). All numbers come from
that document. Output: poster_assets/poster_draft.pdf
"""
import os
import matplotlib
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Spacer, Frame, Image, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "poster_draft.pdf")

ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Sans", os.path.join(ttf, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-B", os.path.join(ttf, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-O", os.path.join(ttf, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(ttf, "DejaVuSansMono.ttf")))

CARD = colors.HexColor("#8C1515")
INK = colors.HexColor("#1f2430")
SLATE = colors.HexColor("#48505e")
LINE = colors.HexColor("#c9d0db")
PANEL = colors.HexColor("#f5f7fa")
BLUE = colors.HexColor("#2a7de1")
GREEN = colors.HexColor("#2e8b57")
GREENBG = colors.HexColor("#e8f5ee")

W, H = 36 * inch, 24 * inch


def S(name, **kw):
    base = dict(fontName="Sans", fontSize=18, leading=22.5, textColor=INK,
                spaceAfter=8, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


st_h = S("h", fontName="Sans-B", fontSize=27, leading=30, textColor=CARD,
         spaceBefore=15, spaceAfter=9)
st_body = S("body", fontSize=18, leading=22.5, spaceAfter=9)
st_bul = S("bul", fontSize=17.5, leading=22, leftIndent=15, spaceAfter=6)
st_small = S("small", fontSize=14.5, leading=18, textColor=SLATE, spaceAfter=7)
st_cap = S("cap", fontName="Sans-O", fontSize=13, leading=16, textColor=SLATE,
           alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)


def bullets(items, style=st_bul):
    return [Paragraph("<bullet>&#8226;</bullet>&nbsp; " + t, style) for t in items]


# =========================== HEADER ===========================
def draw_header(c):
    """Draw header; return the y (from bottom) of the bottom rule."""
    line1 = "Self-Generated Curriculum via Joint Conjecturer/Prover RLOO"
    line2 = "for Arithmetic Reasoning"
    maxw = W - 2.0 * inch
    ts = 72
    while pdfmetrics.stringWidth(line1, "Sans-B", ts) > maxw and ts > 40:
        ts -= 1
    y1 = H - 1.05 * inch
    c.setFillColor(INK)
    c.setFont("Sans-B", ts)
    c.drawCentredString(W / 2, y1, line1)
    y2 = y1 - 1.14 * ts
    c.drawCentredString(W / 2, y2, line2)
    ya = y2 - 0.46 * inch
    c.setFont("Sans", 23)
    c.setFillColor(SLATE)
    c.drawCentredString(W / 2, ya,
                        "Finn Stäblein, Nicholas Allen   •   Stanford CS224R, Spring 2026   •   Qwen 2.5-0.5B on Countdown")
    band_h = 1.3 * inch
    band_bottom = ya - 0.26 * inch - band_h
    c.setFillColor(PANEL)
    c.setStrokeColor(CARD)
    c.setLineWidth(2)
    c.roundRect(0.5 * inch, band_bottom, W - 1.0 * inch, band_h, 8, stroke=1, fill=1)
    take = Paragraph(
        '<b>A small (0.5B) LLM with a deterministic verifier can train its own curriculum</b> — yielding '
        '<b>+111% relative pass@1 over SFT</b> with <b>~6× less compute than vanilla RLOO</b>, '
        'while preserving sample diversity.',
        S("take", fontSize=26, leading=32, alignment=TA_CENTER))
    fr = Frame(1.2 * inch, band_bottom, W - 2.4 * inch, band_h, showBoundary=0,
               topPadding=10, bottomPadding=6, leftPadding=8, rightPadding=8)
    if fr.addFromList([take], c):
        print("WARNING: takeaway band overflow")
    rule_y = band_bottom - 0.16 * inch
    c.setStrokeColor(CARD)
    c.setLineWidth(3)
    c.line(0.5 * inch, rule_y, W - 0.5 * inch, rule_y)
    return rule_y


# =========================== TABLES ===========================
def config_table(width):
    head = ["Run", "C reward", "Mix", "Update samples", "Notes"]
    rows = [
        ["v6 (joint)", "binary band", "0% C / fixed", "~400", "minimal-viable joint training"],
        ["v9 (joint)", "shaped tent", "50% / 50%", "~13,000", "+ length penalty, conservative C hyperparams"],
        ["RLOO (single-policy)", "n/a", "100% fixed", "~80,000", "vanilla RLOO baseline"],
    ]
    data = [head] + rows
    cw = [w * width for w in (0.20, 0.14, 0.13, 0.16, 0.37)]
    t = Table(data, colWidths=cw, hAlign="LEFT")
    ts = [
        ("FONT", (0, 0), (-1, 0), "Sans-B", 13),
        ("FONT", (0, 1), (-1, -1), "Sans", 12.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
        ("GRID", (0, 0), (-1, -1), 0.7, LINE),
        ("ALIGN", (1, 0), (3, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        # highlight v9 (ours) row
        ("BACKGROUND", (0, 2), (-1, 2), GREENBG),
        ("FONT", (0, 2), (0, 2), "Sans-B", 12.5),
        ("TEXTCOLOR", (0, 2), (0, 2), GREEN),
    ]
    t.setStyle(TableStyle(ts))
    return t


def results_table(width):
    head = ["Model", "pass@1", "pass@4", "pass@8", "pass@16"]
    rows = [
        ["SFT baseline", "0.244", "0.534", "0.669", "0.760"],
        ["RLOO baseline", "0.547", "0.694", "0.718", "0.740"],
        ["Joint v6 step 16", "0.345", "0.651", "0.748", "0.800"],
        ["Joint v9 latest", "0.515", "0.654", "0.704", "0.760"],
    ]
    data = [head] + rows
    cw = [w * width for w in (0.34, 0.165, 0.165, 0.165, 0.165)]
    t = Table(data, colWidths=cw, hAlign="LEFT")
    ts = [
        ("FONT", (0, 0), (-1, 0), "Sans-B", 14.5),
        ("FONT", (0, 1), (-1, -1), "Sans", 14.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
        ("GRID", (0, 0), (-1, -1), 0.7, LINE),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        # highlight v9 (ours) row
        ("BACKGROUND", (0, 4), (-1, 4), GREENBG),
        ("FONT", (0, 4), (-1, 4), "Sans-B", 14.5),
        ("TEXTCOLOR", (0, 4), (0, 4), GREEN),
        # bold the per-column best cells
        ("FONT", (1, 2), (1, 2), "Sans-B", 14.5),  # pass@1 best = RLOO
        ("FONT", (2, 2), (2, 2), "Sans-B", 14.5),  # pass@4 best = RLOO
        ("FONT", (3, 3), (3, 3), "Sans-B", 14.5),  # pass@8 best = v6
        ("FONT", (4, 3), (4, 3), "Sans-B", 14.5),  # pass@16 best = v6
        ("TEXTCOLOR", (1, 2), (2, 2), CARD),
        ("TEXTCOLOR", (3, 3), (4, 3), BLUE),
    ]
    t.setStyle(TableStyle(ts))
    return t


def two_figs(width):
    from PIL import Image as PILImage
    gap = 0.25 * inch
    fw = (width - gap) / 2

    def im(path):
        iw, ih = PILImage.open(path).size
        return Image(path, width=fw, height=fw * ih / iw)
    cap_a = Paragraph("Figure 2. pass@k by model (fair eval, 2048 tokens).", st_cap)
    cap_b = Paragraph("Figure 3. pass@1 / pass@4 across v6 training steps.", st_cap)
    inner = Table([[im(os.path.join(HERE, "headline_passk_fair.png")),
                    im(os.path.join(HERE, "training_curve.png"))],
                   [cap_a, cap_b]],
                  colWidths=[fw, fw])
    inner.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, 0), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
    ]))
    return inner


def conclusion_box(width):
    inner = Paragraph(
        "<b>Joint conjecturer/prover RLOO works for arithmetic reasoning with a rule-based verifier.</b> "
        "v9 more than doubles SFT pass@1 while matching vanilla RLOO at ~6× less compute. Both joint "
        "variants (v6, v9) preserve sample diversity on pass@16, whereas vanilla RLOO collapses it. The "
        "principles transfer: shaped reward, distributional anchoring via mixed data, length penalty, and "
        "conservative C-side regularization are necessary for stable two-policy joint training.",
        S("concl", fontSize=17, leading=21.5))
    t = Table([[inner]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ec")),
        ("BOX", (0, 0), (-1, -1), 2.2, CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    return t


def conjecturer_table(width):
    head = ["Conjecturer model + prompt", "Parseable", "Solvable"]
    rows = [
        ["Solve-SFT + chat prompt", "25%", "0%"],
        ["Base Qwen + completion prefill", "85%", "35%"],
        ["Solve-SFT + completion prefill", "95%", "45%"],
        ["C-SFT (mini-trained) + chat", "100%", "0%"],
    ]
    data = [head] + rows
    cw = [w * width for w in (0.56, 0.22, 0.22)]
    t = Table(data, colWidths=cw, hAlign="LEFT")
    ts = [
        ("FONT", (0, 0), (-1, 0), "Sans-B", 13),
        ("FONT", (0, 1), (-1, -1), "Sans", 13),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
        ("GRID", (0, 0), (-1, -1), 0.7, LINE),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        # chosen config (row 3): Solve-SFT + completion prefill
        ("BACKGROUND", (0, 3), (-1, 3), GREENBG),
        ("FONT", (0, 3), (-1, 3), "Sans-B", 13),
        ("TEXTCOLOR", (0, 3), (0, 3), GREEN),
        # flag the 0%-solvable rows in cardinal
        ("TEXTCOLOR", (2, 1), (2, 1), CARD),
        ("FONT", (2, 1), (2, 1), "Sans-B", 13),
        ("TEXTCOLOR", (2, 4), (2, 4), CARD),
        ("FONT", (2, 4), (2, 4), "Sans-B", 13),
    ]
    t.setStyle(TableStyle(ts))
    return t


def fig1(width):
    from PIL import Image as PILImage
    p = os.path.join(HERE, "method_diagram.png")
    iw, ih = PILImage.open(p).size
    return [Image(p, width=width, height=width * ih / iw),
            Paragraph("Figure 1. One training step of the joint conjecturer (blue) / prover (orange) loop.",
                      st_cap)]


# =========================== BUILD ===========================
c = canvas.Canvas(OUT, pagesize=(W, H))
rule_y = draw_header(c)

MARGIN = 0.5 * inch
GAP = 0.4 * inch
usable = W - 2 * MARGIN
COL_W = (usable - 2 * GAP) / 3
body_top = rule_y - 0.18 * inch
body_bot = MARGIN
col_h = body_top - body_bot
col_x = [MARGIN, MARGIN + COL_W + GAP, MARGIN + 2 * (COL_W + GAP)]
IWIDTH = COL_W - 0.3 * inch

# ---------- Column 1 ----------
c1 = []
c1 += [Paragraph("1. Problem", st_h)]
c1 += [Paragraph("LLM reasoning training is bottlenecked by static curricula. Verifiable tasks "
                 "(math, code) let us score outputs at training time — can we use that to "
                 "<b>generate our own problems</b> at the model's frontier of ability?", st_body)]
c1 += [Paragraph("We test this on <b>Countdown</b>: given numbers <font name='Mono'>n</font> "
                 "(e.g. [3, 4, 6, 8]) and a target <font name='Mono'>T</font> (e.g. 24), produce an "
                 "arithmetic expression using +, −, ×, ÷ that uses each number exactly once and "
                 "equals T. A rule-based Python verifier scores correctness.", st_body)]
c1 += [Paragraph("2. Related Work", st_h)]
c1 += bullets([
    "<b>DeepSeek-R1-Zero, TinyZero (2025):</b> RL from a rule-based verifier yields strong reasoning — but on <i>static</i> data.",
    "<b>Dong et al.:</b> conjecturer/prover for <i>formal theorem proving</i> with a proof checker.",
    "<b>Subramaniam et al.:</b> multi-model co-evolution on fixed problems.",
    "<b>Kirk et al. (2024):</b> documents diversity loss as a known RLHF failure.",
])
c1 += [Paragraph("<b>Our novel contribution:</b> first conjecturer/prover joint training applied to "
                 "<b>arithmetic reasoning with a rule-based verifier</b>, plus a brute-force solvability "
                 "filter and a shaped difficulty-band reward.", st_body)]
c1 += [Paragraph("3. Method", st_h)]
c1 += [Paragraph("Two LLMs — same Qwen 2.5 0.5B architecture, both warm-started from SFT. Each "
                 "training step runs a 9-phase loop (Figure 1).", st_body)]
c1 += [Paragraph("<b>Rewards (v9).</b> &nbsp;P: verifier score (0 / 0.1 / 1.0) minus &alpha;·(len / max_len). "
                 "&nbsp;C (shaped tent): <font name='Mono'>r_C = max(0, 1 − 2.5·|p − 0.4|)</font> — peaks at "
                 "moderate difficulty (solve-rate p = 0.4), zero outside [0, 0.8].", st_body)]
c1 += fig1(IWIDTH)

# ---------- Column 2 ----------
c2 = []
c2 += [Paragraph("4. Experimental Setup", st_h)]
c2 += bullets([
    "<b>Model:</b> Qwen 2.5 0.5B Base → SFT (Asap7772/cog_behav_all_strategies) → joint RLOO.",
    "<b>Test set:</b> 50 held-out problems, 16 samples each at temp 0.6, top_p 0.95, top_k 20.",
    "<b>Eval:</b> vLLM, <b>max_tokens=2048</b> (fair budget — see notes).",
    "<b>Compute:</b> single H100 via Modal; v9 trained 100 steps in ~3–4 hr.",
])
c2 += [Paragraph("5. Configurations", st_h)]
c2 += [config_table(IWIDTH), Spacer(1, 10)]
c2 += [Paragraph("6. Results", st_h)]
c2 += [Paragraph("Fair eval at max_tokens=2048, n=50 test problems, k=16 samples. "
                 "Per-column best in color; <font color='#2e8b57'><b>v9 = ours</b></font>.", st_small)]
c2 += [results_table(IWIDTH), Spacer(1, 6)]
c2 += [Paragraph("7. Figures", st_h)]
c2 += [two_figs(IWIDTH)]
c2 += [Paragraph("8. Conjecturer Ablation: Prompt &#215; Init", st_h)]
c2 += [Paragraph("Getting C to emit problems that are both <b>parseable</b> and <b>solvable</b> depends on "
                 "prompt mode and initialization. Completion-mode prompting with a "
                 "<font name='Mono'>&lt;problem&gt; numbers=[</font> prefill suppresses the SFT solve-reflex:",
                 st_body)]
c2 += [conjecturer_table(IWIDTH), Spacer(1, 7)]
c2 += [Paragraph("<b>Key insight:</b> C-SFT reached 100% parseable but 0% solvable — it learned the "
                 "<i>format</i> but not the arithmetic relationship between the numbers and the target. "
                 "Solve-SFT carries an implicit solvability prior from thousands of complete solution traces.",
                 st_body)]

# ---------- Column 3 ----------
c3 = []
c3 += [Paragraph("9. Key Findings", st_h)]
c3 += [Paragraph("<b>F1 — Joint curriculum matches vanilla RLOO at ~6× less compute.</b> v9 reaches "
                 "pass@1 = 0.515 with ~13,000 update samples; RLOO needs ~80,000 to reach 0.547 — "
                 "comparable performance at 6× the cost. Adaptive problem generation uses each step more "
                 "efficiently.", st_body)]
c3 += [Paragraph("<b>F2 — Joint methods preserve diversity; vanilla RLOO does not.</b> After ~80,000 "
                 "samples RLOO's pass@16 = 0.740, still below the untrained SFT baseline (0.760). Both "
                 "joint variants match or exceed it (v6 = 0.800, v9 = 0.760). The curriculum acts as an "
                 "implicit regularizer against over-specialization (cf. Kirk et al. 2024).", st_body)]
c3 += [Paragraph("<b>F3 — The joint regime is narrow but stable when properly designed.</b> v9 needed: "
                 "(a) shaped C reward, (b) 50% mixed fixed/curriculum data, (c) length penalty &alpha;=0.1, "
                 "(d) conservative C hyperparams (c_lr=1e-6, c_kl=0.02). Without these, two-policy joint RL "
                 "collapses — C's gradient dies or it drifts off the SFT distribution.", st_body)]
c3 += [Paragraph("10. Methodological Notes", st_h)]
c3 += bullets([
    "<b>Eval token budget matters.</b> max_tokens=1024 silently truncated long chains-of-thought before they emit &lt;answer&gt;, disproportionately penalizing more-trained models. All numbers here use 2048.",
    "<b>n=50, 16 samples/problem.</b> Bootstrap CIs are wide; treat differences &lt;0.05 absolute as suggestive, not conclusive.",
])
c3 += [Paragraph("11. Limitations", st_h)]
c3 += bullets([
    "n=50 test set, single seed — wide CIs.",
    "0.5B model — diversity collapse is sharper than at larger scale.",
    "Solver filter accepts non-integer intermediates (inherits verifier true-division semantics).",
    "C-SFT side experiment: 720 (prompt, problem) pairs → 100% parseable but 0% solvable (surface patterns without task constraints).",
])
c3 += [Paragraph("12. Future Work", st_h)]
c3 += bullets([
    "Adaptive difficulty band that shifts as the prover improves (cf. Dong et al.).",
    "Held-out eval during training for early stopping at peak.",
    "Larger model (3B–7B) — curriculum gains may strengthen at scale.",
    "Dedicated SFT corpus for C with a verifier-augmented loss penalizing unsolvable outputs.",
])
c3 += [Spacer(1, 4),
       Paragraph("<b>References.</b> DeepSeek-AI 2025 (R1); Pan et al. (TinyZero); Dong &amp; Ma "
                 "(self-play, formal proofs); Subramaniam et al. (co-evolution); Kirk et al. 2024 (RLHF diversity).",
                 st_small)]
c3 += [Spacer(1, 8), conclusion_box(COL_W - 0.3 * inch)]

# ---------- flow into frames ----------
for i, story in enumerate([c1, c2, c3]):
    fr = Frame(col_x[i], body_bot, COL_W, col_h, showBoundary=0,
               topPadding=6, bottomPadding=6, leftPadding=14, rightPadding=14)
    remaining = list(story)            # addFromList pops drawn items in place
    fr.addFromList(remaining, c)
    print(f"column {i+1}: " + ("OK" if not remaining else f"OVERFLOW — {len(remaining)} flowables dropped"))

c.showPage()
c.save()
print("saved", OUT, "| size in:", W / inch, "x", H / inch)
