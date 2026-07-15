from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "CBSWEG2_MCO3_Automation_Test_Plan.pdf"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
PALE_BLUE = colors.HexColor("#D9EAF7")
PALE_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#6B7280")
GRID = colors.HexColor("#B8C2CC")
WHITE = colors.white
BLACK = colors.HexColor("#1F2933")


def register_fonts():
    candidates = [
        ("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"),
        ("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for name, path in candidates:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))


register_fonts()
FONT = "Arial" if "Arial" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
BOLD = "Arial-Bold" if "Arial-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "TitleCustom", parent=styles["Title"], fontName=BOLD, fontSize=21,
    leading=25, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8,
)
subtitle_style = ParagraphStyle(
    "SubtitleCustom", parent=styles["Normal"], fontName=FONT, fontSize=9.5,
    leading=13, textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=16,
)
h1_style = ParagraphStyle(
    "H1Custom", parent=styles["Heading1"], fontName=BOLD, fontSize=14,
    leading=17, textColor=NAVY, spaceBefore=4, spaceAfter=8,
)
body_style = ParagraphStyle(
    "BodyCustom", parent=styles["BodyText"], fontName=FONT, fontSize=9,
    leading=12, textColor=BLACK, spaceAfter=5,
)
small_style = ParagraphStyle(
    "SmallCustom", parent=body_style, fontSize=7.5, leading=9.5, spaceAfter=0,
)
small_center = ParagraphStyle(
    "SmallCenter", parent=small_style, alignment=TA_CENTER,
)
header_style = ParagraphStyle(
    "HeaderCell", parent=small_style, fontName=BOLD, textColor=WHITE,
    alignment=TA_CENTER, leading=9,
)
module_style = ParagraphStyle(
    "ModuleCell", parent=small_style, fontName=BOLD, textColor=NAVY,
    alignment=TA_LEFT,
)
note_style = ParagraphStyle(
    "NoteCustom", parent=body_style, fontSize=8.5, leading=11,
    textColor=colors.HexColor("#374151"), leftIndent=8, rightIndent=8,
)


def P(text, style=small_style):
    return Paragraph(text, style)


def page_footer(canvas, doc):
    canvas.saveState()
    width, _ = doc.pagesize
    canvas.setStrokeColor(colors.HexColor("#D5DAE0"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.46 * inch, width - doc.rightMargin, 0.46 * inch)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(doc.leftMargin, 0.28 * inch, "CBSWEG2 MCO3 - Automation Test Plan")
    canvas.drawRightString(width - doc.rightMargin, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def info_table():
    rows = [
        [P("Project Name", module_style), P("EDSA Traffic Model: Predictions, Insights, &amp; Classifications", body_style)],
        [P("Project Description", module_style), P(
            "This project prepares an EDSA road-traffic accident classification pipeline using MMDA records from 2007 to 2016. It standardizes and validates the source data, engineers model-ready features, and will generate accident-risk predictions once the CBADVAI classifier is finalized.", body_style)],
        [P("Dataset", module_style), P("RTA_EDSA_2007-2016.csv - Road Traffic Accident Data of EDSA, Metro Manila (2007-2016), sourced from Mendeley Data.", body_style)],
        [P("Team Members", module_style), P("Ed Bennett Borromeo - Project Manager / Scrum Master<br/>Rovick Dompor - QA / Tester<br/>Matthew Fuentes - Full Stack Developer", body_style)],
        [P("GitHub Repository", module_style), P("https://github.com/S180-TK/CBSWEG2", body_style)],
    ]
    t = Table(rows, colWidths=[1.55 * inch, 5.75 * inch], repeatRows=0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


unit_tests = [
    ("EDSA-UT-001", "filter_columns()", "Keeps only the six required variables and removes unrelated columns.", "DataFrame containing the six required columns plus EXTRA_COL", "Only ADDRESS, DATETIME_PST, SEVERITY, X, Y, and COLLISION_TYPE remain."),
    ("EDSA-UT-002", "drop_missing_address()", "Removes records whose ADDRESS is missing.", "['EDSA Guadalupe', None, 'EDSA Cubao']", "Two rows remain and ADDRESS contains no null value."),
    ("EDSA-UT-003", "clean_edsa_address()", "Standardizes northbound/southbound tags and barangay abbreviations.", "'EDSA N.B. Bgy Guadalupe'", "'EDSA (NB) Brgy. Guadalupe'"),
    ("EDSA-UT-004", "clean_edsa_address()", "Expands road-type and city abbreviations.", "'Ortigas Ave. Q.C.'", "'Ortigas Avenue Quezon City.'"),
    ("EDSA-UT-005", "normalize_address_column()", "Trims whitespace and applies consistent capitalization before address cleaning.", "'  edsa guadalupe  '", "'Edsa Guadalupe'"),
    ("EDSA-UT-006", "extract_hour()", "Parses DATETIME_PST, creates HOUR, and removes the original datetime column.", "'2016-05-01 14:30:00'", "HOUR = 14; DATETIME_PST is removed."),
    ("EDSA-UT-007", "find_invalid_range_rows()", "Returns numeric values outside an accepted inclusive range.", "HOUR = [-1, 5, 23, 24]; valid range = 0 to 23", "Rows containing -1 and 24 are returned."),
    ("EDSA-UT-008", "add_severity_num()", "Maps severity labels to ordinal values used in the CBDATSI analysis.", "['Property', 'Injury', 'Fatal']", "[1, 2, 3]"),
    ("EDSA-UT-009", "add_hour_bin()", "Groups accident hours into four time-of-day categories.", "[0, 7, 13, 19]", "['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']"),
]


def unit_table():
    data = [
        [P("Module", header_style), P("Data Cleaning and Feature Engineering", header_style), "", "", ""],
        [P("ID", header_style), P("Function", header_style), P("Test Description", header_style), P("Input", header_style), P("Expected Output", header_style)],
    ]
    for rid, fn, desc, inp, out in unit_tests:
        data.append([P(rid, small_center), P(fn, module_style), P(desc), P(inp), P(out)])
    t = Table(data, colWidths=[0.83*inch, 1.25*inch, 2.48*inch, 2.35*inch, 2.59*inch], repeatRows=2)
    style = [
        ("SPAN", (1, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, 1), BLUE),
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row in range(2, len(data)):
        if row % 2 == 0:
            style.append(("BACKGROUND", (0, row), (-1, row), PALE_GRAY))
    t.setStyle(TableStyle(style))
    return t


def flow_box(label, width=1.48*inch):
    t = Table([[P(label, ParagraphStyle("Flow", parent=small_center, fontName=BOLD, textColor=NAVY, leading=10))]], colWidths=[width], rowHeights=[0.58*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.9, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def system_flow():
    arrow = P("&#8594;", ParagraphStyle("Arrow", parent=small_center, fontName=BOLD, fontSize=15, textColor=BLUE))
    row1 = [flow_box("Input record / dataset"), arrow, flow_box("Input validation"), arrow, flow_box("Inference preprocessing"), arrow, flow_box("Trained classifier")]
    row2 = ["", "", "", "", "", "", P("&#8595;", ParagraphStyle("Down", parent=small_center, fontName=BOLD, fontSize=15, textColor=BLUE))]
    row3 = [flow_box("Displayed prediction"), P("&#8592;", ParagraphStyle("ArrowL", parent=small_center, fontName=BOLD, fontSize=15, textColor=BLUE)), flow_box("Output formatting"), P("&#8592;", ParagraphStyle("ArrowL2", parent=small_center, fontName=BOLD, fontSize=15, textColor=BLUE)), flow_box("Predicted class"), "", ""]
    t = Table([row1, row2, row3], colWidths=[1.48*inch, .3*inch, 1.48*inch, .3*inch, 1.48*inch, .3*inch, 1.48*inch], rowHeights=[.64*inch, .3*inch, .64*inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return t


def metrics_table():
    rows = [
        [P("Metric", header_style), P("Formula / Representation", header_style), P("Purpose", header_style), P("Target", header_style)],
        [P("Accuracy", module_style), P("Correct predictions / Total predictions"), P("Overall proportion of correct classifications."), P("Higher than the majority-class baseline.")],
        [P("Precision", module_style), P("TP / (TP + FP), reported per class and as a macro average"), P("Measures the reliability of positive predictions."), P("Report per class; monitor the minority class.")],
        [P("Recall", module_style), P("TP / (TP + FN), reported per class and as a macro average"), P("Measures how many actual cases in each class are detected."), P("Prioritize recall for the casualty/minority class.")],
        [P("F1-score", module_style), P("2 x (Precision x Recall) / (Precision + Recall)"), P("Balances precision and recall under class imbalance."), P("Macro F1 higher than the baseline model.")],
        [P("Confusion Matrix", module_style), P("Actual class by predicted class count matrix"), P("Shows which classes the model confuses."), P("The casualty/minority class must not be consistently missed.")],
    ]
    t = Table(rows, colWidths=[1.15*inch, 2.05*inch, 2.25*inch, 2.2*inch], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for row in (2, 4):
        style.append(("BACKGROUND", (0, row), (-1, row), PALE_GRAY))
    t.setStyle(TableStyle(style))
    return t


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    portrait = PageTemplate(
        id="portrait",
        pagesize=letter,
        frames=[Frame(0.62*inch, 0.58*inch, 7.26*inch, 9.86*inch, id="portrait-frame")],
        onPage=page_footer,
    )
    wide = PageTemplate(
        id="wide",
        pagesize=landscape(letter),
        frames=[Frame(0.48*inch, 0.58*inch, 10.04*inch, 7.36*inch, id="wide-frame")],
        onPage=page_footer,
    )
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=0.62*inch, rightMargin=0.62*inch,
        topMargin=0.58*inch, bottomMargin=0.58*inch,
        title="CBSWEG2 MCO3: Automation Test Plan",
        author="CBSWEG2 Team",
    )
    doc.addPageTemplates([portrait, wide])
    story = [
        Spacer(1, 0.08*inch),
        Paragraph("CBSWEG2 MCO3", subtitle_style),
        Paragraph("Automation Test Plan", title_style),
        Paragraph("EDSA Traffic Model: Predictions, Insights, &amp; Classifications", subtitle_style),
        Paragraph("Project Information", h1_style),
        info_table(),
        Spacer(1, 0.16*inch),
        Table([[P("Scope note", module_style), P("Unit tests below cover the CBDATSI data-cleaning and feature-engineering functions currently implemented. Model-specific tests will be added after the CBADVAI classifier and prediction target are finalized.", note_style)]], colWidths=[1.05*inch, 6.25*inch], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7E6")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D9A441")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])),
        NextPageTemplate("wide"),
        PageBreak(),
    ]
    story.append(Paragraph("Unit Testing", h1_style))
    story.append(Paragraph("The following cases correspond to the data-processing functions currently implemented in <b>src/models/data_processing.py</b>.", body_style))
    story.append(unit_table())
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())
    story.append(Paragraph("System Testing", h1_style))
    story.append(Paragraph("The final system test will validate the application from input to displayed prediction. The classifier-specific function names will be inserted after the CBADVAI pipeline is finalized.", body_style))
    story.append(Spacer(1, 0.08*inch))
    story.append(system_flow())
    story.append(Spacer(1, 0.17*inch))
    story.append(Table([[P("Excluded from this system flow", module_style), P("Exploratory data analysis, plots, correlation studies, and one-time offline dataset preparation that are not executed during normal inference.", note_style)]], colWidths=[1.7*inch, 5.6*inch], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY), ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])))
    story.append(Spacer(1, 0.24*inch))
    story.append(Paragraph("Performance Evaluation", h1_style))
    story.append(Paragraph("Because the final model is not yet available, targets are stated relative to a baseline rather than as unsupported fixed percentages.", body_style))
    story.append(metrics_table())
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
