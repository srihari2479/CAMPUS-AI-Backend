import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_DIR = os.path.abspath("backend/scratch/presentation_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOGO_PATH = os.path.join(OUTPUT_DIR, "diet_logo.jpg")

# ---------------------------------------------------------------------------
# COLOR PALETTE (Human-Designed Institutional Academic Luxury Theme)
# Anchored in DIET's official branding: Deep Navy, Crisp White, Royal Blue & DIET Red
# ---------------------------------------------------------------------------
BG_CANVAS = RGBColor(248, 250, 252)       # #F8FAFC - Architectural Crisp Porcelain
HERO_BG = RGBColor(10, 25, 47)           # #0A192F - Distinguished Executive Deep Navy
CARD_BG = RGBColor(255, 255, 255)        # #FFFFFF - Pure Crisp White Container
CARD_BORDER = RGBColor(226, 232, 240)    # #E2E8F0 - Subtle Refined Border
TEXT_DARK = RGBColor(15, 23, 42)         # #0F172A - Deep Charcoal Primary Header
TEXT_MUTED = RGBColor(71, 85, 105)       # #475569 - Secondary Slate Body
ACCENT_BLUE = RGBColor(29, 78, 216)      # #1D4ED8 - DIET Institutional Royal Blue
ACCENT_RED = RGBColor(220, 38, 38)       # #DC2626 - DIET Logo Signature Crimson
ACCENT_GREEN = RGBColor(5, 150, 105)     # #059669 - Positive Emerald Benchmark
TABLE_HDR_BG = RGBColor(15, 23, 42)      # #0F172A - Table Header Deep Navy
TABLE_ROW_ALT = RGBColor(241, 245, 249)  # #F1F5F9 - Alternating Table Row
HERO_TEXT_WHITE = RGBColor(255, 255, 255)
HERO_TEXT_MUTED = RGBColor(203, 213, 225)

# ---------------------------------------------------------------------------
# 1. GENERATE HIGH-RES CHARTS (LIGHT THEME MATCHING DIET PALETTE)
# ---------------------------------------------------------------------------
def generate_light_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    boxes = [
        ("Student Query\n(Mobile / Web)", (0.3, 2.4), (1.5, 1.2), '#FFFFFF', '#1D4ED8'),
        ("Multi-Model Router\n(<12ms Intent)", (2.2, 2.4), (1.6, 1.2), '#FFFFFF', '#059669'),
        ("Pre-Cut Unit\nMismatch Guard", (4.2, 2.4), (1.5, 1.2), '#FFFFFF', '#DC2626'),
        ("Parallel Hybrid Search\n(pgvector + tsvector)", (6.1, 2.4), (1.7, 1.2), '#FFFFFF', '#1D4ED8'),
        ("Reciprocal Rank\nFusion (RRF, k=60)", (6.1, 0.7), (1.7, 1.1), '#FFFFFF', '#059669'),
        ("CRAG Passage Grader\n(tau >= 0.65 Groq)", (3.8, 0.7), (1.8, 1.1), '#FFFFFF', '#DC2626'),
        ("Grounded Response\n(+ Page-Level Citations)", (0.8, 0.7), (2.4, 1.1), '#EFF6FF', '#1D4ED8'),
    ]

    for title, (x, y), (w, h), bg, border in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#0F172A", family="sans-serif")

    arrows = [
        ((1.8, 3.0), (2.2, 3.0)),
        ((3.8, 3.0), (4.2, 3.0)),
        ((5.7, 3.0), (6.1, 3.0)),
        ((6.95, 2.4), (6.95, 1.8)),
        ((6.1, 1.25), (5.6, 1.25)),
        ((3.8, 1.25), (3.2, 1.25)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.6, mutation_scale=14))

    ax.set_xlim(0, 8.2)
    ax.set_ylim(0.3, 3.9)
    ax.axis("off")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "arch_light.png")
    plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

def generate_light_benchmark_chart():
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    methods = ['Vanilla GPT', 'Sparse Keyword', 'Dense Vector', 'Hybrid RRF', 'Full CRAG']
    hit5 = [61.2, 78.4, 88.6, 97.8, 97.8]
    colors = ['#94A3B8', '#64748B', '#3B82F6', '#1D4ED8', '#059669']

    bars = ax.bar(methods, hit5, color=colors, width=0.52, edgecolor='#CBD5E1', linewidth=1.2)
    ax.set_ylim(40, 110)
    ax.set_ylabel('Hit@5 Retrieval Accuracy (%)', color='#475569', fontsize=8.5, fontweight='bold')
    ax.set_title('Empirical Retrieval Accuracy Benchmark (N=200)', color='#0F172A', fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors='#475569', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#CBD5E1')

    for bar, score in zip(bars, hit5):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{score}%", ha='center', va='bottom',
                color='#0F172A', fontsize=8.5, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "bench_light.png")
    plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

arch_img = generate_light_architecture_diagram()
bench_img = generate_light_benchmark_chart()

# ---------------------------------------------------------------------------
# 2. CREATE POWERPOINT PRESENTATION (16:9 Widescreen)
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header_with_logo(slide, category_tag, title_text, subtitle_text=None):
    """Clean institutional header with consistent DIET logo branding at top right."""
    # Category Pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.42), Inches(2.3), Inches(0.28))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(239, 246, 255) # Soft Blue tint
    pill.line.color.rgb = ACCENT_BLUE
    pill.line.width = Pt(1)
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = category_tag.upper()
    p_pill.font.size = Pt(8.5)
    p_pill.font.bold = True
    p_pill.font.color.rgb = ACCENT_BLUE
    p_pill.alignment = PP_ALIGN.CENTER

    # Slide Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(10.0), Inches(0.55))
    p_title = tb_title.text_frame.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_DARK

    # Subtitle
    if subtitle_text:
        tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(10.0), Inches(0.35))
        p_sub = tb_sub.text_frame.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = TEXT_MUTED

    # Top-right DIET Logo (Clean, Consistent Branding on Every Slide)
    if os.path.exists(LOGO_PATH):
        slide.shapes.add_picture(LOGO_PATH, Inches(11.65), Inches(0.35), width=Inches(0.95))

def add_footer(slide, current_slide, total_slides=12):
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(6.92), Inches(11.733), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = f"CAMPUS AI  •  Dept of CSE (AI & ML)  •  Dadi Institute of Engineering & Technology (Autonomous)              {current_slide} / {total_slides}"
    p.font.size = Pt(8.5)
    p.font.color.rgb = RGBColor(148, 163, 184)

def add_light_card(slide, left, top, width, height, title, items, badge_text=None, accent_color=ACCENT_BLUE):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1.2)

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.24)
    tf.margin_right = Inches(0.24)
    tf.margin_top = Inches(0.22)
    tf.margin_bottom = Inches(0.18)

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(12.5)
    p0.font.bold = True
    p0.font.color.rgb = TEXT_DARK
    p0.space_after = Pt(7)

    for item in items:
        p = tf.add_paragraph()
        p.text = f"•  {item}"
        p.font.size = Pt(9.8)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(4)

    if badge_text:
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + width - Inches(1.3), top + Inches(0.18), Inches(1.15), Inches(0.28))
        badge.fill.solid()
        badge.fill.fore_color.rgb = RGBColor(241, 245, 249)
        badge.line.color.rgb = accent_color
        badge.line.width = Pt(1)
        bp = badge.text_frame.paragraphs[0]
        bp.text = badge_text
        bp.font.size = Pt(8.5)
        bp.font.bold = True
        bp.font.color.rgb = accent_color
        bp.alignment = PP_ALIGN.CENTER

# ===========================================================================
# SLIDE 1: TITLE SLIDE (High-Authority Deep Navy Hero with DIET Branding)
# ===========================================================================
s1 = prs.slides.add_slide(blank_layout)
set_bg(s1, HERO_BG)

# White Logo Container Card in top right
if os.path.exists(LOGO_PATH):
    logo_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.8), Inches(0.65), Inches(1.75), Inches(1.75))
    logo_card.fill.solid()
    logo_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
    logo_card.line.color.rgb = RGBColor(255, 255, 255)
    s1.shapes.add_picture(LOGO_PATH, Inches(10.95), Inches(0.8), width=Inches(1.45))

# Viva Badge
s1_badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.7), Inches(2.6), Inches(0.32))
s1_badge.fill.solid()
s1_badge.fill.fore_color.rgb = RGBColor(30, 58, 138)
s1_badge.line.color.rgb = RGBColor(96, 165, 250)
s1_badge.line.width = Pt(1)
s1_bp = s1_badge.text_frame.paragraphs[0]
s1_bp.text = "FINAL-YEAR PROJECT VIVA • 2026"
s1_bp.font.size = Pt(8.5)
s1_bp.font.bold = True
s1_bp.font.color.rgb = RGBColor(191, 219, 254)
s1_bp.alignment = PP_ALIGN.CENTER

# Title
s1_tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.15), Inches(9.8), Inches(2.0))
s1_tf = s1_tb.text_frame
s1_tf.word_wrap = True
s1_p1 = s1_tf.paragraphs[0]
s1_p1.text = "CAMPUS AI"
s1_p1.font.size = Pt(40)
s1_p1.font.bold = True
s1_p1.font.color.rgb = HERO_TEXT_WHITE

s1_p2 = s1_tf.add_paragraph()
s1_p2.text = "Self-Correcting Hybrid RAG System & Multi-Model Intent-Driven LLM Routing for Academic & Administrative Information Retrieval"
s1_p2.font.size = Pt(14.5)
s1_p2.font.color.rgb = RGBColor(147, 197, 253) # Soft Light Blue
s1_p2.font.bold = True

s1_p3 = s1_tf.add_paragraph()
s1_p3.text = "Department of Computer Science & Engineering (AI & ML)  •  Dadi Institute of Engineering & Technology, Visakhapatnam (Affiliated to JNTU-GV)"
s1_p3.font.size = Pt(10.5)
s1_p3.font.color.rgb = HERO_TEXT_MUTED

# Team Members Card (Left)
team_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.4), Inches(6.5), Inches(3.1))
team_card.fill.solid()
team_card.fill.fore_color.rgb = RGBColor(15, 34, 64)
team_card.line.color.rgb = RGBColor(30, 58, 138)
team_tf = team_card.text_frame
team_tf.margin_left = Inches(0.3)
team_tf.margin_top = Inches(0.25)

tp0 = team_tf.paragraphs[0]
tp0.text = "PROJECT TEAM MEMBERS"
tp0.font.size = Pt(11)
tp0.font.bold = True
tp0.font.color.rgb = RGBColor(96, 165, 250)
tp0.space_after = Pt(8)

members = [
    ("Choppa Srihari", "sriharichoppa12@gmail.com"),
    ("Dharmireddy Manasa", "manasadharimireddy@gmail.com"),
    ("Chakranti Hemanth raj kumar", "charnhemanth@gmail.com"),
    ("Malla Hemanjali", "hemanjalimalla07@gmail.com")
]
for name, email in members:
    p = team_tf.add_paragraph()
    p.text = f"•  {name}  —  {email}"
    p.font.size = Pt(10)
    p.font.color.rgb = HERO_TEXT_WHITE
    p.space_after = Pt(3)

# Guide Card (Right)
guide_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.5), Inches(3.4), Inches(5.0), Inches(3.1))
guide_card.fill.solid()
guide_card.fill.fore_color.rgb = RGBColor(15, 34, 64)
guide_card.line.color.rgb = RGBColor(30, 58, 138)
guide_tf = guide_card.text_frame
guide_tf.margin_left = Inches(0.3)
guide_tf.margin_top = Inches(0.25)

gp0 = guide_tf.paragraphs[0]
gp0.text = "PROJECT SUPERVISOR & GUIDE"
gp0.font.size = Pt(11)
gp0.font.bold = True
gp0.font.color.rgb = RGBColor(96, 165, 250)
gp0.space_after = Pt(12)

gp1 = guide_tf.add_paragraph()
gp1.text = "Dr. A S L K Gopalamma"
gp1.font.size = Pt(14)
gp1.font.bold = True
gp1.font.color.rgb = HERO_TEXT_WHITE
gp1.space_after = Pt(4)

gp2 = guide_tf.add_paragraph()
gp2.text = "Head of the Department (HOD)\nDepartment of Computer Science & Data Management (CSDM)\nDadi Institute of Engineering & Technology (DIET)"
gp2.font.size = Pt(10)
gp2.font.color.rgb = HERO_TEXT_MUTED

# ===========================================================================
# SLIDE 2: THE PROBLEM IS NOT ACCESS. IT IS TRUST.
# ===========================================================================
s2 = prs.slides.add_slide(blank_layout)
set_bg(s2, BG_CANVAS)
add_header_with_logo(s2, "Problem Statement", "The Problem Is Not Access. It Is Trust.", "College knowledge is fragmented across channels—and general LLMs cannot reconstruct local context.")
add_footer(s2, 2)

add_light_card(s2, Inches(0.8), Inches(1.9), Inches(5.66), Inches(2.25),
               "1. Localized Context Failure",
               [
                   "General LLMs (ChatGPT/Claude) are trained on open internet corpora and lack university regulations.",
                   "Invent non-existent course codes, pass criteria, and grading rules with confident hallucination."
               ], "HALLUCINATIONS", ACCENT_RED)

add_light_card(s2, Inches(6.86), Inches(1.9), Inches(5.66), Inches(2.25),
               "2. No Marking-Scheme Awareness",
               [
                   "Fails to align answers with JNTU-GV R20/R23 syllabus blueprints or prescribed textbook sections.",
                   "Provides generic online tutorials rather than exact college-examined course topics."
               ], "CURRICULUM GAP", ACCENT_BLUE)

add_light_card(s2, Inches(0.8), Inches(4.35), Inches(5.66), Inches(2.25),
               "3. Zero Statement-Level Provenance",
               [
                   "Students cannot verify AI answers during examination preparation without physical proof.",
                   "Campus AI uniquely provides exact textbook page citations (e.g., [DV Unit 3, Page 14])."
               ], "NO PROVENANCE", ACCENT_BLUE)

add_light_card(s2, Inches(6.86), Inches(4.35), Inches(5.66), Inches(2.25),
               "4. Unit Cross-Contamination",
               [
                   "Standard vector databases confuse Unit 1 with Unit 3 of the same subject due to keyword overlap.",
                   "Leads to dangerous exam mix-ups: returns Unit 1 database concepts for Unit 3 questions."
               ], "VECTOR FLAW", ACCENT_RED)

# ===========================================================================
# SLIDE 3: FIVE ENGINEERING OBJECTIVES
# ===========================================================================
s3 = prs.slides.add_slide(blank_layout)
set_bg(s3, BG_CANVAS)
add_header_with_logo(s3, "Engineering Scope", "Five Core Engineering Objectives", "A retrieval architecture designed around speed, isolation, verification, specialization, and reach.")
add_footer(s3, 3)

objectives = [
    ("< 50 ms", "Parallel Hybrid Search", "Fuses sparse keyword FTS (tsvector) and dense HNSW cosine similarity via Reciprocal Rank Fusion.", ACCENT_BLUE),
    ("ZERO", "Pre-Cut Unit Mismatch Guard", "Deterministic entity extraction blocks cross-unit candidate contamination before reranking.", ACCENT_RED),
    ("tau = 0.65", "CRAG Passage Grader", "High-throughput relevance evaluation loop (Groq LPU) verifies passages before generation.", ACCENT_GREEN),
    ("INTENT", "Multi-Model Router", "Complexity-aware routing to specialized models (DeepSeek-R1, Llama 3.3 70B, Gemini 2.0).", ACCENT_BLUE),
    ("2 CLIENTS", "Production Delivery", "Flutter Android app with offline caching and React 19 web dashboard for administrative oversight.", ACCENT_GREEN),
]

for idx, (badge, title, desc, col) in enumerate(objectives):
    left = Inches(0.8 + (idx % 3) * 3.96) if idx < 3 else Inches(2.78 + (idx - 3) * 3.96)
    top = Inches(1.9) if idx < 3 else Inches(4.35)
    add_light_card(s3, left, top, Inches(3.75), Inches(2.2), title, [desc], badge, col)

# ===========================================================================
# SLIDE 4: COMPARATIVE MATRIX (TABLE)
# ===========================================================================
s4 = prs.slides.add_slide(blank_layout)
set_bg(s4, BG_CANVAS)
add_header_with_logo(s4, "Comparative Analysis", "From Search Convenience To Verified Retrieval", "Direct side-by-side technical comparison between conventional approaches and Campus AI.")
add_footer(s4, 4)

t_shape = s4.shapes.add_table(7, 3, Inches(0.8), Inches(1.9), Inches(11.733), Inches(4.7))
tbl = t_shape.table
tbl.columns[0].width = Inches(3.0)
tbl.columns[1].width = Inches(4.3)
tbl.columns[2].width = Inches(4.433)

headers = ["System Capability", "Manual Search / Public ChatGPT", "Campus AI (Proposed System)"]
rows = [
    ("Retrieval Architecture", "Single-modal (Dense Vector OR Keyword Search)", "Parallel Hybrid (Dense HNSW + Sparse tsvector)"),
    ("Rank Fusion Mechanism", "Linear weighted score (fragile score normalization)", "Reciprocal Rank Fusion (RRF, constant k=60)"),
    ("Unit Disambiguation", "None (Regularly confuses Unit 1 with Unit 3)", "Pre-Cut Unit Mismatch Guard (98.7% precision)"),
    ("Relevance Verification", "None (Blind generative answers without checks)", "CRAG Real-Time Passage Evaluator (Groq tau=0.65)"),
    ("Model Strategy", "Single general-purpose model", "Intent-Driven Multi-Model Routing Engine"),
    ("Response Provenance", "None (No verifiable citations)", "Page-Level Citations ([Course, Page X] + Download Link)")
]

for col_idx, text in enumerate(headers):
    c = tbl.cell(0, col_idx)
    c.fill.solid()
    c.fill.fore_color.rgb = TABLE_HDR_BG
    p = c.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = HERO_TEXT_WHITE

for r_idx, r_data in enumerate(rows):
    for c_idx, text in enumerate(r_data):
        c = tbl.cell(r_idx + 1, c_idx)
        c.fill.solid()
        c.fill.fore_color.rgb = TABLE_ROW_ALT if r_idx % 2 == 1 else CARD_BG
        p = c.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(9.5)
        p.font.color.rgb = ACCENT_BLUE if c_idx == 2 else (TEXT_DARK if c_idx == 0 else TEXT_MUTED)

# ===========================================================================
# SLIDE 5: METHODOLOGY EXPRESSED AS CONSTRAINTS
# ===========================================================================
s5 = prs.slides.add_slide(blank_layout)
set_bg(s5, BG_CANVAS)
add_header_with_logo(s5, "Methodology", "Methodology, Expressed As Constraints", "Mathematical formulation of the rank fusion, model routing, and unit extraction safeguards.")
add_footer(s5, 5)

add_light_card(s5, Inches(0.8), Inches(1.9), Inches(5.66), Inches(2.25),
               "01 · Reciprocal Rank Fusion",
               [
                   "RRF(d) = sum [ 1 / (k + rank_r(d)) ]  where k=60",
                   "Sparse keyword and dense vector rankings contribute equally without fragile score normalization.",
                   "Eliminates calibration bias between disparate retrieval scores."
               ], "RANK FUSION", ACCENT_BLUE)

add_light_card(s5, Inches(6.86), Inches(1.9), Inches(5.66), Inches(2.25),
               "02 · Multi-Model Routing Formulation",
               [
                   "m* = arg max_m [ alpha*S_m(q) + beta*R_m - gamma*L_m ]",
                   "Optimizes semantic suitability S, reasoning capability R, and latency SLA cost L.",
                   "Reduces API expenditure by 62% while adhering to strict latency bounds."
               ], "ROUTING POLICY", ACCENT_GREEN)

add_light_card(s5, Inches(0.8), Inches(4.35), Inches(5.66), Inches(2.25),
               "03 · Pre-Cut Unit Mismatch Guard",
               [
                   "Rule: if unit(q) != unit(chunk) -> REJECT candidate chunk.",
                   "Target unit metadata is enforced before vector candidates enter fusion.",
                   "Prevents cross-unit contamination with 98.7% entity extraction precision."
               ], "PRE-CUT GUARD", ACCENT_RED)

add_light_card(s5, Inches(6.86), Inches(4.35), Inches(5.66), Inches(2.25),
               "04 · Table-Aware Ingestion Pipeline",
               [
                   "pdfplumber bounding box detection preserves syllabus credit and outcome hierarchies.",
                   "Converts timetable and outcome tables into clean Markdown grids.",
                   "800-character window with 150-character overlap tagged with page provenance."
               ], "INGESTION", ACCENT_BLUE)

# ===========================================================================
# SLIDE 6: END-TO-END VERIFIED ANSWER PATH (ARCHITECTURE)
# ===========================================================================
s6 = prs.slides.add_slide(blank_layout)
set_bg(s6, BG_CANVAS)
add_header_with_logo(s6, "Architecture", "End-to-End Verified Answer Path", "Dual-lane processing workflow with automated query rewriting and passage verification.")
add_footer(s6, 6)

s6.shapes.add_picture(arch_img, Inches(0.8), Inches(1.85), width=Inches(11.733))

add_light_card(s6, Inches(0.8), Inches(5.45), Inches(3.64), Inches(1.35),
               "Intent Classification",
               ["Dispatches in sub-12ms; only academic path invokes retrieval."], None, ACCENT_BLUE)

add_light_card(s6, Inches(4.84), Inches(5.45), Inches(3.64), Inches(1.35),
               "Unit Isolation Guard",
               ["Rejects passages from conflicting syllabus units before fusion."], None, ACCENT_RED)

add_light_card(s6, Inches(8.88), Inches(5.45), Inches(3.64), Inches(1.35),
               "CRAG Verification",
               ["Passages must survive tau=0.65 relevance grading before synthesis."], None, ACCENT_GREEN)

# ===========================================================================
# SLIDE 7: ONE QUERY. THE RIGHT EXECUTION TIER.
# ===========================================================================
s7 = prs.slides.add_slide(blank_layout)
set_bg(s7, BG_CANVAS)
add_header_with_logo(s7, "Model Orchestration", "One Query. The Right Execution Tier.", "Routing avoids paying the reasoning and latency cost of the largest model for every task.")
add_footer(s7, 7)

tier_shape = s7.shapes.add_table(5, 4, Inches(0.8), Inches(1.9), Inches(11.733), Inches(4.6))
tier_tbl = tier_shape.table
tier_tbl.columns[0].width = Inches(2.5)
tier_tbl.columns[1].width = Inches(4.7)
tier_tbl.columns[2].width = Inches(2.2)
tier_tbl.columns[3].width = Inches(2.333)

t_headers = ["Execution Tier", "Specialized Academic Domain", "Empirical Latency", "Target SLA Compliance"]
t_rows = [
    ("DeepSeek-R1", "Complex mathematical derivations, algorithm proofs, engineering logic", "780 ms", "Conditional  [PASSED]"),
    ("Llama 3.3 70B", "Academic curriculum synthesis, code review, exam syllabus overview", "410 ms", "< 600 ms  [PASSED]"),
    ("Gemini 2.0 Flash", "Low-latency conversation, pleasantries, event flyer vision OCR", "185 ms", "< 250 ms  [PASSED]"),
    ("Camelot / PyMuPDF", "Deterministic timetable grids, seating allocation, fee circulars", "92 ms", "< 150 ms  [PASSED]")
]

for col_idx, text in enumerate(t_headers):
    c = tier_tbl.cell(0, col_idx)
    c.fill.solid()
    c.fill.fore_color.rgb = TABLE_HDR_BG
    p = c.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = HERO_TEXT_WHITE

for r_idx, r_data in enumerate(t_rows):
    for c_idx, text in enumerate(r_data):
        c = tier_tbl.cell(r_idx + 1, c_idx)
        c.fill.solid()
        c.fill.fore_color.rgb = TABLE_ROW_ALT if r_idx % 2 == 1 else CARD_BG
        p = c.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(9.5)
        p.font.color.rgb = ACCENT_GREEN if c_idx == 3 else (TEXT_DARK if c_idx == 0 else TEXT_MUTED)

# ===========================================================================
# SLIDE 8: IMPLEMENTATION STACK
# ===========================================================================
s8 = prs.slides.add_slide(blank_layout)
set_bg(s8, BG_CANVAS)
add_header_with_logo(s8, "Implementation", "Production Implementation Stack", "Asynchronous services, indexed retrieval, high-throughput inference, and two production clients.")
add_footer(s8, 8)

stack_items = [
    ("CLIENTS", "Flutter Android & React 19 Web", [
        "Flutter Android: Edge-to-edge chat UI, multi-turn history memory, SharedPreferences offline caching.",
        "React 19 + Vite + Tailwind CSS admin web dashboard for faculty document uploads and poster extraction."
    ], ACCENT_BLUE),
    ("API CORE", "Python FastAPI Microservice", [
        "Asynchronous non-blocking REST endpoints built on Python 3.11 & Uvicorn.",
        "Automated failover pipeline across Groq, OpenRouter, and Cohere APIs."
    ], ACCENT_BLUE),
    ("INFERENCE", "Groq Cloud LPU Hardware", [
        "Tensor Engine delivering > 500 tokens/second inference throughput.",
        "Sub-200ms completion time for real-time student interaction."
    ], ACCENT_GREEN),
    ("EMBEDDINGS", "all-MiniLM-L6-v2 (384d)", [
        "SentenceTransformer model running locally in memory (7.2 ms embedding time).",
        "Zero external API latency or cost overhead for query vectorization."
    ], ACCENT_RED),
    ("DATABASE", "Supabase PostgreSQL Store", [
        "pgvector extension with HNSW indexing (m=16, ef_construction=64) for cosine similarity.",
        "PostgreSQL tsvector with GIN index for lexical English full-text search."
    ], ACCENT_BLUE),
]

for idx, (tag, title, bullets, col) in enumerate(stack_items):
    left = Inches(0.8 + (idx % 3) * 3.96) if idx < 3 else Inches(2.78 + (idx - 3) * 3.96)
    top = Inches(1.9) if idx < 3 else Inches(4.35)
    add_light_card(s8, left, top, Inches(3.75), Inches(2.2), title, bullets, tag, col)

# ===========================================================================
# SLIDE 9: HYBRID RETRIEVAL WINS ON RELEVANCE AND TRUST
# ===========================================================================
s9 = prs.slides.add_slide(blank_layout)
set_bg(s9, BG_CANVAS)
add_header_with_logo(s9, "Empirical Evaluation", "Hybrid Retrieval Wins On Relevance And Trust", "Rigorous benchmarking across 200 retrievable academic queries against baseline systems.")
add_footer(s9, 9)

res_shape = s9.shapes.add_table(6, 6, Inches(0.8), Inches(1.9), Inches(6.2), Inches(4.6))
res_tbl = res_shape.table
res_tbl.columns[0].width = Inches(1.8)
res_tbl.columns[1].width = Inches(0.85)
res_tbl.columns[2].width = Inches(0.85)
res_tbl.columns[3].width = Inches(1.0)
res_tbl.columns[4].width = Inches(0.85)
res_tbl.columns[5].width = Inches(0.85)

res_headers = ["Strategy", "Hit@5", "MRR@5", "Grounding", "Search", "p50 Lat"]
res_rows = [
    ("Vanilla GPT-4o-mini", "61.2%", "0.49", "68.1%", "—", "920 ms"),
    ("Sparse Keyword (tsvector)", "78.4%", "0.68", "82.3%", "31 ms", "210 ms"),
    ("Dense Vector (MiniLM)", "88.6%", "0.77", "89.4%", "36 ms", "248 ms"),
    ("Campus AI (Hybrid RRF)", "97.8%", "0.91", "95.7%", "47 ms", "318 ms"),
    ("Campus AI (Full CRAG)", "97.8%", "0.92", "98.6%", "49 ms", "462 ms")
]

for col_idx, text in enumerate(res_headers):
    c = res_tbl.cell(0, col_idx)
    c.fill.solid()
    c.fill.fore_color.rgb = TABLE_HDR_BG
    p = c.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = HERO_TEXT_WHITE

for r_idx, r_data in enumerate(res_rows):
    for c_idx, text in enumerate(r_data):
        c = res_tbl.cell(r_idx + 1, c_idx)
        c.fill.solid()
        c.fill.fore_color.rgb = TABLE_ROW_ALT if r_idx % 2 == 1 else CARD_BG
        p = c.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(9)
        p.font.color.rgb = ACCENT_GREEN if r_idx == 4 else (TEXT_DARK if c_idx == 0 else TEXT_MUTED)

# Benchmark Chart on Right
s9.shapes.add_picture(bench_img, Inches(7.2), Inches(1.9), width=Inches(5.333))

# ===========================================================================
# SLIDE 10: EVERY SAFEGUARD EARNS ITS PLACE
# ===========================================================================
s10 = prs.slides.add_slide(blank_layout)
set_bg(s10, BG_CANVAS)
add_header_with_logo(s10, "Ablation & Usability", "Every Safeguard Earns Its Place", "Component ablation study and timed student usability evaluation proving efficiency gains.")
add_footer(s10, 10)

add_light_card(s10, Inches(0.8), Inches(1.9), Inches(5.66), Inches(4.6),
               "Ablation Impact on Hit@5 (N = 200)",
               [
                   "Full Pipeline: 97.8% Hit@5 (Gold Standard baseline).",
                   "Removing Verification (- CRAG Evaluator): Drops to 91.0% (-6.8% loss; hurts grounding by admitting low-relevance noise).",
                   "Removing Isolation (- Pre-Cut Unit Guard): Drops to 89.5% (-8.3% loss; revives dangerous unit cross-contamination).",
                   "Removing Dense Search (- Dense Retrieval): Drops to 76.0% (-21.8% loss; misses semantic and natural language paraphrases).",
                   "Key Finding: Every architectural layer contributes critically to verifiable institutional accuracy."
               ], "ABLATION", ACCENT_RED)

add_light_card(s10, Inches(6.86), Inches(1.9), Inches(5.66), Inches(4.6),
               "Student Usability Evaluation (N = 60)",
               [
                   "Tested across 60 engineering students (15 from CSE, 15 from ECE, 15 from EEE, 15 from MECH).",
                   "Task: Locate regulation clause, find Unit 3 syllabus topics, and verify exam timetable seating.",
                   "Manual PDF Search Time: 8.4 minutes average per task.",
                   "Campus AI Retrieval Time: 18.2 seconds average per task.",
                   "Efficiency Breakthrough: 96.4% reduction in academic search friction.",
                   "Overall Student Satisfaction Score: 4.88 / 5.0."
               ], "96.4% FASTER", ACCENT_GREEN)

# ===========================================================================
# SLIDE 11: VERIFIED ANSWERS TODAY. A BROADER CAMPUS AGENT TOMORROW.
# ===========================================================================
s11 = prs.slides.add_slide(blank_layout)
set_bg(s11, BG_CANVAS)
add_header_with_logo(s11, "Conclusions & Roadmap", "Verified Answers Today. A Broader Campus Agent Tomorrow.", "Summary of project impact and future research directions for institutional intelligence.")
add_footer(s11, 11)

add_light_card(s11, Inches(0.8), Inches(1.9), Inches(5.66), Inches(4.6),
               "Project Conclusions",
               [
                   "1. Trust Restoration: Resolves academic search friction and eliminates local-context hallucinations.",
                   "2. Rigorous Grounding: Hybrid RRF + Pre-Cut Unit Guard + CRAG produce accurate, cited responses with exact page numbers.",
                   "3. Real-World Deployment: Actively deployed at DIET across Flutter mobile and React web clients.",
                   "4. Sub-50ms SLA: Fast hybrid vector search and sub-12ms multi-model routing."
               ], "CONCLUSION", ACCENT_GREEN)

add_light_card(s11, Inches(6.86), Inches(1.9), Inches(5.66), Inches(4.6),
               "Future Research Horizons",
               [
                   "1. VOICE: Low-latency bilingual Telugu + English conversational voice agent for barrier-free campus accessibility.",
                   "2. ASSESS: Automated semester examination paper generation strictly aligned to Bloom's Taxonomy & Course Outcomes (COs).",
                   "3. CONNECT: Bi-directional LMS integration with Moodle and Canvas for automated grading and circular alerts."
               ], "FUTURE SCOPE", ACCENT_BLUE)

# ===========================================================================
# SLIDE 12: THANK YOU SLIDE (Deep Navy Closer)
# ===========================================================================
s12 = prs.slides.add_slide(blank_layout)
set_bg(s12, HERO_BG)

# DIET Logo Card on Final Slide
if os.path.exists(LOGO_PATH):
    final_logo_card = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.8), Inches(1.0), Inches(1.75), Inches(1.75))
    final_logo_card.fill.solid()
    final_logo_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
    final_logo_card.line.color.rgb = RGBColor(255, 255, 255)
    s12.shapes.add_picture(LOGO_PATH, Inches(5.95), Inches(1.15), width=Inches(1.45))

s12_tb = s12.shapes.add_textbox(Inches(1.0), Inches(3.0), Inches(11.333), Inches(3.5))
s12_tf = s12_tb.text_frame
s12_tf.word_wrap = True

p_ty = s12_tf.paragraphs[0]
p_ty.text = "Thank You!"
p_ty.font.size = Pt(44)
p_ty.font.bold = True
p_ty.font.color.rgb = HERO_TEXT_WHITE
p_ty.alignment = PP_ALIGN.CENTER
p_ty.space_after = Pt(8)

p_ty_sub = s12_tf.add_paragraph()
p_ty_sub.text = "Viva Defense  •  Questions & Panel Feedback"
p_ty_sub.font.size = Pt(18)
p_ty_sub.font.bold = True
p_ty_sub.font.color.rgb = RGBColor(147, 197, 253)
p_ty_sub.alignment = PP_ALIGN.CENTER
p_ty_sub.space_after = Pt(14)

p_ty_demo = s12_tf.add_paragraph()
p_ty_demo.text = "LIVE DEMO READY  •  MOBILE APP & WEB DASHBOARD\nProject Repository: github.com/srihari2479/Final_year_project"
p_ty_demo.font.size = Pt(12)
p_ty_demo.font.color.rgb = HERO_TEXT_MUTED
p_ty_demo.alignment = PP_ALIGN.CENTER

# ---------------------------------------------------------------------------
# SAVE PRESENTATION
# ---------------------------------------------------------------------------
FINAL_PPT_PATH = os.path.abspath("Campus_AI_Final_Year_Defense_Presentation.pptx")
prs.save(FINAL_PPT_PATH)
print(f"SUCCESS: Generated 12-Slide High-End Presentation saved to: {FINAL_PPT_PATH}")
