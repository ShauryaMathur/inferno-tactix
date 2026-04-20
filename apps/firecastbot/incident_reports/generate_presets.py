"""
Regenerate incident_report_low.pdf and incident_report_high.pdf
using the same visual template as incident_report_boulder.pdf.

Run from repo root:
    python apps/firecastbot/incident_reports/generate_presets.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------


def make_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=base["Normal"],
        fontSize=16,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    section = ParagraphStyle(
        "SectionHeader",
        parent=base["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
        spaceBefore=14,
        spaceAfter=6,
        leftIndent=6,
    )
    body = ParagraphStyle(
        "BodyText",
        parent=base["Normal"],
        fontSize=10,
        fontName="Helvetica",
        leading=14,
        leftIndent=6,
        rightIndent=6,
        spaceAfter=8,
    )
    return title, section, body


TABLE_STYLE = TableStyle(
    [
        # header row
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        # body rows
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        # grid
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
        # padding
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
)

COL_WIDTHS = [2.6 * inch, 4.2 * inch]

# Cell paragraph styles (set once, reused per table cell so text wraps)
_CELL_STYLES: dict = {}


def _cell_style(bold: bool = False) -> ParagraphStyle:
    key = "bold" if bold else "normal"
    if key not in _CELL_STYLES:
        _CELL_STYLES[key] = ParagraphStyle(
            f"Cell_{key}",
            fontSize=10,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            leading=13,
            spaceAfter=0,
            spaceBefore=0,
        )
    return _CELL_STYLES[key]


def _cell(text: str, bold: bool = False) -> Paragraph:
    """Wrap a table cell string in a Paragraph so it word-wraps."""
    return Paragraph(text, _cell_style(bold))


def make_table(raw_rows: list[list[str]]) -> Table:
    """Convert plain-string rows to Paragraph-wrapped cells, then build Table."""
    para_rows = []
    for i, row in enumerate(raw_rows):
        is_header = i == 0
        para_rows.append([_cell(cell, bold=is_header) for cell in row])
    t = Table(para_rows, colWidths=COL_WIDTHS)
    t.setStyle(TABLE_STYLE)
    return t


def build_doc(path: Path, elements):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )
    doc.build(elements)
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# LOW risk — Acton / Soledad Canyon (content preserved exactly)
# ---------------------------------------------------------------------------


def generate_low():
    title_style, section_style, body_style = make_styles()
    path = OUT_DIR / "incident_report_low.pdf"

    elems = []

    elems.append(Paragraph("FIRE THREAT ASSESSMENT REPORT", title_style))

    elems.append(Paragraph("INCIDENT OVERVIEW", section_style))
    elems.append(
        make_table(
            [
                ["Field", "Value"],
                ["Date", "January 10, 2025"],
                ["Region", "Acton and Soledad Canyon Corridor, Los Angeles County, CA"],
                ["Approx Area", "395 acres"],
                ["Overall Risk Level", "LOW"],
            ]
        )
    )

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("ENVIRONMENTAL & FIRE BEHAVIOR INPUTS", section_style))
    elems.append(
        make_table(
            [
                ["Parameter", "Value"],
                ["Elevation", "2,600–3,600 ft"],
                ["Slope", "8–18 degrees (rolling foothills)"],
                ["Land Cover", "Light chaparral and annual grass with broken brush continuity"],
                ["Fuel Model", "FM 1/5 mixed (grass and light brush)"],
                ["Fuel Load", "2–5 tons/acre"],
                ["Dead Fuel Moisture (1h/10h/100h)", "9% / 10% / 12%"],
                ["Live Fuel Moisture", "95%"],
                ["Air Temperature", "63 °F"],
                ["Relative Humidity", "38%"],
                ["Wind Speed", "6–12 mph (light gusts to 18 mph)"],
                ["Wind Direction", "NE to E"],
                ["Recent Precipitation", "0.7 in (last 7 days)"],
                ["Drought Index (ERC)", "41 (Moderate)"],
                ["Canopy Cover", "10–25%"],
                ["Crown Base Height", "N/A to sparse shrubs"],
                ["Fuel Continuity", "Low to moderate (broken brush and road interruptions)"],
                ["Aspect", "South and east aspects with local sheltered draws"],
            ]
        )
    )

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("DERIVED FIRE BEHAVIOR METRICS (SIMULATED)", section_style))
    elems.append(
        make_table(
            [
                ["Metric", "Value"],
                ["Direction of Maximum Spread", "E to SE along light wind alignment and drainages"],
                ["Rate of Spread (ROS)", "0.15–0.35 m/s (Low)"],
                ["Flame Length", "0.5–1.2 m"],
                ["Fireline Intensity", "150–450 kW/m"],
                ["Crown Fire Potential", "Low"],
                ["Spotting Potential", "Low (under 0.1 km)"],
                ["Containment Probability (Initial Attack)", "0.89 (HIGH)"],
                ["Time to Containment (Simulated)", "40–70 steps"],
                ["Average Burnt Area", "400–900 cells"],
                ["Suppression Feasibility", "HIGH for direct attack and perimeter hold"],
                ["Evacuation Urgency", "LOW"],
            ]
        )
    )

    elems.append(Spacer(1, 12))
    elems.append(Paragraph("RL AGENT RECOMMENDATION", section_style))
    elems.append(
        Paragraph(
            "Use direct attack supported by road access and short hose lays. Prioritize quick edge "
            "securement, mop-up near access corridors, and holding the heel and flanks before any "
            "afternoon wind increase.",
            body_style,
        )
    )

    elems.append(Paragraph("OPERATIONAL IMPLICATIONS", section_style))
    elems.append(
        Paragraph(
            "This incident behaves like a manageable initial attack fire with limited spotting and low "
            "resistance to control. Preparation should focus on containment efficiency, maintaining "
            "lookout coverage in drainages, and preventing rekindles near cured grass patches.",
            body_style,
        )
    )

    build_doc(path, elems)


# ---------------------------------------------------------------------------
# HIGH risk — Pacific Palisades / Santa Monica Mountains (content preserved exactly)
# ---------------------------------------------------------------------------


def generate_high():
    title_style, section_style, body_style = make_styles()
    path = OUT_DIR / "incident_report_high.pdf"

    elems = []

    elems.append(Paragraph("FIRE THREAT ASSESSMENT REPORT", title_style))

    elems.append(Paragraph("INCIDENT OVERVIEW", section_style))
    elems.append(
        make_table(
            [
                ["Field", "Value"],
                ["Date", "January 11, 2025"],
                [
                    "Region",
                    "Pacific Palisades and Santa Monica Mountains Interface, Los Angeles County, CA",
                ],
                ["Approx Area", "23,654 acres"],
                ["Overall Risk Level", "HIGH"],
            ]
        )
    )

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("ENVIRONMENTAL & FIRE BEHAVIOR INPUTS", section_style))
    elems.append(
        make_table(
            [
                ["Parameter", "Value"],
                ["Elevation", "400–1,900 ft"],
                ["Slope", "18–38 degrees (canyons and ridgelines)"],
                [
                    "Land Cover",
                    "Coastal sage scrub, chaparral, ornamental fuels, and dense WUI structure exposure",
                ],
                ["Fuel Model", "FM 4/6 mixed (heavy brush and timber litter pockets)"],
                ["Fuel Load", "8–14 tons/acre"],
                ["Dead Fuel Moisture (1h/10h/100h)", "4% / 5% / 7%"],
                ["Live Fuel Moisture", "68%"],
                ["Air Temperature", "78 °F"],
                ["Relative Humidity", "11%"],
                ["Wind Speed", "22–35 mph (gusts to 45 mph)"],
                ["Wind Direction", "N to NE Santa Ana pattern"],
                ["Recent Precipitation", "Trace to 0.1 in (last 14 days)"],
                ["Drought Index (ERC)", "83 (Very High)"],
                ["Canopy Cover", "20–55%"],
                ["Crown Base Height", "3–10 ft in brush/tree transition zones"],
                ["Fuel Continuity", "High (continuous brush with heavy ember receptive exposure)"],
                [
                    "Aspect",
                    "Wind-exposed south and west slopes, aligned canyons, and ridge saddles",
                ],
            ]
        )
    )

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("DERIVED FIRE BEHAVIOR METRICS (SIMULATED)", section_style))
    elems.append(
        make_table(
            [
                ["Metric", "Value"],
                [
                    "Direction of Maximum Spread",
                    "S to SW downslope and down-canyon with long-range ember transport",
                ],
                ["Rate of Spread (ROS)", "1.8–3.4 m/s (High)"],
                ["Flame Length", "4–9 m"],
                ["Fireline Intensity", "4,500–11,000 kW/m"],
                ["Crown Fire Potential", "High (active torching and short crown runs possible)"],
                ["Spotting Potential", "High (0.8–2.5 km)"],
                ["Containment Probability (Initial Attack)", "0.18 (LOW)"],
                ["Time to Containment (Simulated)", "220–320 steps"],
                ["Average Burnt Area", "42,000–68,000 cells"],
                [
                    "Suppression Feasibility",
                    "LOW to MODERATE except at hardened structure defense locations",
                ],
                ["Evacuation Urgency", "HIGH"],
            ]
        )
    )

    elems.append(Spacer(1, 12))
    elems.append(Paragraph("RL AGENT RECOMMENDATION", section_style))
    elems.append(
        Paragraph(
            "Favor point protection, structure triage, and indirect control anchored to major barriers. "
            "Avoid committing crews to exposed head-fire positions during peak wind periods. Prioritize "
            "life safety, evacuation support, and protection of critical infrastructure corridors.",
            body_style,
        )
    )

    elems.append(Paragraph("OPERATIONAL IMPLICATIONS", section_style))
    elems.append(
        Paragraph(
            "Expect fast-moving, wind-driven fire with severe spotting and repeated alignment runs "
            "through canyons and WUI edges. Preparation must emphasize conservative tactics, surge "
            "resource staging, contingency lines, and rapid adjustment to Santa Ana wind shifts.",
            body_style,
        )
    )

    build_doc(path, elems)


if __name__ == "__main__":
    generate_low()
    generate_high()
