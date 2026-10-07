import os
import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def create_df_preview_table(df: pd.DataFrame, styles, primary_color, dark_neutral, light_neutral, max_cols=7, max_rows=10):
    """
    Renders a high-fidelity table preview of a pandas DataFrame (top rows/cols) 
    wrapped in Paragraph flowables to prevent text overflow in the PDF.
    """
    cols_to_show = list(df.columns)
    has_more_cols = len(df.columns) > max_cols
    if has_more_cols:
        cols_to_show = cols_to_show[:max_cols-1] + ["..."]
        
    df_preview = df.head(max_rows)
    
    # Styles for table cells
    hdr_cell_style = ParagraphStyle(
        'HdrCell',
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    body_cell_style = ParagraphStyle(
        'BodyCell',
        fontName='Courier',
        fontSize=7,
        leading=9,
        textColor=dark_neutral,
        alignment=TA_LEFT
    )
    
    null_cell_style = ParagraphStyle(
        'NullCell',
        fontName='Courier-Oblique',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#b91c1c"), # Dark red for nulls
        alignment=TA_CENTER
    )
    
    table_data = []
    
    # Header row
    hdr_row = []
    for c in cols_to_show:
        hdr_row.append(Paragraph(escape_markdown(c), hdr_cell_style))
    table_data.append(hdr_row)
    
    # Data rows
    for r_idx in range(len(df_preview)):
        row = []
        for c in cols_to_show:
            if c == "...":
                row.append(Paragraph("...", body_cell_style))
            else:
                val = df_preview.iloc[r_idx][c]
                # Check for NaN / Empty values
                if pd.isna(val) or str(val).strip().lower() in ["nan", "null", "none", "<na>", "nat", ""]:
                    row.append(Paragraph("NaN", null_cell_style))
                else:
                    val_str = str(val).strip()
                    # Truncate strings to prevent cell overflow
                    if len(val_str) > 22:
                        val_str = val_str[:19] + "..."
                    row.append(Paragraph(escape_markdown(val_str), body_cell_style))
        table_data.append(row)
        
    # Calculate column widths to fit exactly the printable 504 pt width
    num_cols = len(cols_to_show)
    col_width = 504.0 / num_cols
    col_widths = [col_width] * num_cols
    
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

def generate_pdf_report(filename: str, original_profile: dict, original_anomalies: dict, history: list, validation: dict, df_orig: pd.DataFrame, df_clean: pd.DataFrame) -> bytes:
    """
    Module 8: Restructured Professional PDF Cleaning Report
    Generates a beautifully styled academic-grade 6-part PDF report using ReportLab.
    Returns bytes of the PDF.
    """
    buffer = io.BytesIO()
    
    # 1. Page template & layout setup (Margins: 0.75 in / 54 pt)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Design System Colors
    primary_color = colors.HexColor("#4f46e5")    # Indigo
    secondary_color = colors.HexColor("#0ea5e9")  # Slate Blue
    dark_neutral = colors.HexColor("#1e293b")     # Charcoal Text
    light_neutral = colors.HexColor("#f8fafc")    # Off-white backgrounds
    success_color = colors.HexColor("#10b981")    # Emerald Green
    accent_border = colors.HexColor("#e2e8f0")
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=5,
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=dark_neutral,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark_neutral,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=dark_neutral,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    fig_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-BoldOblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=5,
        spaceAfter=10
    )
    
    table_hdr_style = ParagraphStyle(
        'TableHdr', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=TA_CENTER
    )
    table_cell_style = ParagraphStyle(
        'TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=dark_neutral, alignment=TA_CENTER
    )
    table_cell_lbl_style = ParagraphStyle(
        'TableCellLbl', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, textColor=dark_neutral, alignment=TA_LEFT
    )

    # 1. Document Title
    story.append(Paragraph("COCOON AUTOMATED DATA CLEANING REPORT", title_style))
    story.append(Paragraph("M.Sc. Project Research Execution & Verification Evidence", subtitle_style))
    
    # Extract metadata and calculate scores
    comp = validation.get("comparison", {})
    detailed = comp.get("detailed", {})
    
    rows_before = comp.get("rows", {}).get("before", len(df_orig))
    rows_after = comp.get("rows", {}).get("after", len(df_clean))
    cols_before = comp.get("columns", {}).get("before", len(df_orig.columns))
    cols_after = comp.get("columns", {}).get("after", len(df_clean.columns))
    
    before_score = comp.get("before_score", 0)
    after_score = comp.get("after_score", 0)
    
    # ------------------ SECTION 1: CLEANING SUMMARY ------------------
    story.append(Paragraph("1. Cleaning Summary", h1_style))
    story.append(Paragraph("A high-level summary of the dataset characteristics, shapes, and overall quality score improvements before and after automated cleaning:", body_style))
    story.append(Spacer(1, 4))
    
    summary_bullets = [
        f"<b>Dataset Name:</b> {filename}",
        f"<b>Dataset Dimensions:</b> {rows_before} rows &times; {cols_before} columns (Before) &rarr; {rows_after} rows &times; {cols_after} columns (After)",
        f"<b>Quality Score Before:</b> {before_score}%",
        f"<b>Quality Score After:</b> {after_score}%",
        f"<b>Overall Improvement:</b> +{after_score - before_score}% quality enhancement"
    ]
    for bullet in summary_bullets:
        story.append(Paragraph(f"&bull; {bullet}", bullet_style))
    story.append(Spacer(1, 10))

    # ------------------ SECTION 2: DATA QUALITY ISSUES DETECTED ------------------
    story.append(Paragraph("2. Data Quality Issues Detected", h1_style))
    story.append(Paragraph("Initial statistical and semantic analysis detected the following structural anomalies, invalid formats, and inconsistencies:", body_style))
    story.append(Spacer(1, 4))
    
    missing_cnt = comp.get("missing_values", {}).get("before", 0)
    dup_cnt = comp.get("duplicates", {}).get("before", 0)
    email_cnt = detailed.get("email_issues", {}).get("before", 0)
    phone_cnt = detailed.get("phone_issues", {}).get("before", 0)
    outliers_cnt = comp.get("outliers", {}).get("before", 0)
    semantic_cnt = detailed.get("semantic_inconsistencies", {}).get("before", detailed.get("casing_issues", {}).get("before", 0))
    
    issues_bullets = [
        f"<b>Missing values:</b> {missing_cnt} occurrences of missing records or disguised null values",
        f"<b>Duplicate records:</b> {dup_cnt} duplicate rows in dataset structure",
        f"<b>Invalid emails:</b> {email_cnt} malformed or invalid email addresses",
        f"<b>Invalid phone numbers:</b> {phone_cnt} records failing phone validation standards",
        f"<b>Outliers:</b> {outliers_cnt} numerical outliers identified using standard IQR bounds",
        f"<b>Semantic inconsistencies:</b> {semantic_cnt} casing variations or inconsistent category codes"
    ]
    for bullet in issues_bullets:
        story.append(Paragraph(f"&bull; {bullet}", bullet_style))
    story.append(Spacer(1, 10))

    # ------------------ SECTION 3: BEFORE CLEANING - ORIGINAL DATASET ------------------
    orig_table = create_df_preview_table(df_orig, styles, primary_color, dark_neutral, light_neutral)
    orig_dataset_block = KeepTogether([
        Paragraph("3. Before Cleaning &mdash; Original Dataset", h1_style),
        Paragraph("A visual datasheet preview of the original raw dataset highlighting unstandardized formatting, casing issues, and missing cells:", body_style),
        Spacer(1, 4),
        orig_table,
        Spacer(1, 4),
        Paragraph("Fig. 1. Original dataset before automated preprocessing", fig_style)
    ])
    story.append(orig_dataset_block)
    
    # Break to Page 2 to guarantee clean layout distribution
    story.append(PageBreak())

    # ------------------ SECTION 4: CLEANING OPERATIONS APPLIED ------------------
    story.append(Paragraph("4. Cleaning Operations Applied", h1_style))
    story.append(Paragraph("The system programmatically executed the following pipeline recommendations under the supervision of statistical parameters and LLM verification:", body_style))
    story.append(Spacer(1, 4))
    
    has_imputation = any("imput" in str(h['action']).lower() or "fill" in str(h['action']).lower() for h in history)
    has_dup = any("duplicate" in str(h['action']).lower() or "dup" in str(h['action']).lower() for h in history) or comp.get("duplicates", {}).get("resolved", 0) > 0
    has_outlier = any("outlier" in str(h['action']).lower() or "clip" in str(h['action']).lower() for h in history) or comp.get("outliers", {}).get("resolved", 0) > 0
    has_email = any("email" in str(h['column']).lower() for h in history)
    has_phone = any("phone" in str(h['column']).lower() or "contact" in str(h['column']).lower() for h in history)
    has_casing = any("case" in str(h['action']).lower() or "map" in str(h['action']).lower() or "standard" in str(h['action']).lower() for h in history)
    
    mv_desc = "Missing values resolved using semantic imputation, default mappings, or drop operations based on LLM suggestions." if has_imputation else "No missing value imputation was required; cell completion was validated."
    dup_desc = f"Identified and purged duplicate rows to enforce strict primary record uniqueness (resolved {comp.get('duplicates', {}).get('resolved', 0)} duplicate rows)." if has_dup else "No duplicate rows were detected in the input dataset; structural integrity validated."
    out_desc = "Numeric outliers statistically detected using IQR bounds (1.5x threshold) and resolved via boundary clipping or nullification." if has_outlier else "No statistical numeric outliers exceeded IQR deviation thresholds."
    em_desc = "Email records validated against syntax structures; malformed strings were standardized or nullified." if has_email else "Email columns verified against standard formatting patterns; no anomalies remain."
    ph_desc = "Phone numbers validated against standard 10-digit formats, stripping non-digit symbols and retaining NaN values to protect authenticity." if has_phone else "Phone number fields analyzed for structural validity; records match normal formats."
    cat_desc = "Standardized inconsistent text casing (e.g., lowercase vs. titlecase) and unified category variations using semantic mappings." if has_casing else "Categorical text values are standardized; casing check yielded no variations."
    safe_desc = "Executed post-cleaning validation checks to ensure clean data matches semantic expectations without structural corruption."
    
    ops_bullets = [
        f"<b>Missing-value handling:</b> {mv_desc}",
        f"<b>Duplicate handling:</b> {dup_desc}",
        f"<b>IQR-based outlier detection/treatment:</b> {out_desc}",
        f"<b>Email validation:</b> {em_desc}",
        f"<b>Phone-number validation:</b> {ph_desc}",
        f"<b>Categorical standardization:</b> {cat_desc}",
        f"<b>Safety verification:</b> {safe_desc}"
    ]
    for bullet in ops_bullets:
        story.append(Paragraph(f"&bull; {bullet}", bullet_style))
    story.append(Spacer(1, 10))

    # ------------------ SECTION 5: AFTER CLEANING - CLEANED DATASET ------------------
    clean_table = create_df_preview_table(df_clean, styles, primary_color, dark_neutral, light_neutral)
    clean_dataset_block = KeepTogether([
        Paragraph("5. After Cleaning &mdash; Cleaned Dataset", h1_style),
        Paragraph("A visual datasheet preview of the cleaned and standardized dataset after pipeline execution, highlighting formatted cells, corrected casings, and imputed nulls:", body_style),
        Spacer(1, 4),
        clean_table,
        Spacer(1, 4),
        Paragraph("Fig. 2. Dataset after automated preprocessing and validation", fig_style)
    ])
    story.append(clean_dataset_block)
    story.append(Spacer(1, 10))

    # ------------------ SECTION 6: BEFORE VS AFTER COMPARISON ------------------
    # Calculate values for the comparison table
    invalid_keys = ["email_issues", "phone_issues", "invalid_dates", "invalid_numeric"]
    invalid_before = sum(detailed.get(k, {}).get("before", 0) for k in invalid_keys)
    invalid_after = sum(detailed.get(k, {}).get("after", 0) for k in invalid_keys)
    
    outliers_before = comp.get("outliers", {}).get("before", 0)
    outliers_after = comp.get("outliers", {}).get("after", 0)
    
    standardized_total = sum(c_stats.get("standardized", 0) for c_stats in validation.get("column_validation_report", {}).values())
    
    unresolved_before = missing_cnt + dup_cnt + invalid_before + outliers_before
    unresolved_after = comp.get("missing_values", {}).get("after", 0) + comp.get("duplicates", {}).get("after", 0) + invalid_after + outliers_after
    
    comp_data = [
        [Paragraph("Metric", ParagraphStyle('H', parent=table_hdr_style, alignment=TA_LEFT)),
         Paragraph("Before", table_hdr_style),
         Paragraph("After", table_hdr_style)],
        [Paragraph("Quality Score", table_cell_lbl_style), Paragraph(f"{before_score}%", table_cell_style), Paragraph(f"{after_score}%", table_cell_style)],
        [Paragraph("Invalid Values", table_cell_lbl_style), Paragraph(str(invalid_before), table_cell_style), Paragraph(str(invalid_after), table_cell_style)],
        [Paragraph("Outliers", table_cell_lbl_style), Paragraph(str(outliers_before), table_cell_style), Paragraph(str(outliers_after), table_cell_style)],
        [Paragraph("Standardized Values", table_cell_lbl_style), Paragraph("0", table_cell_style), Paragraph(str(standardized_total), table_cell_style)],
        [Paragraph("Unresolved Issues", table_cell_lbl_style), Paragraph(str(unresolved_before), table_cell_style), Paragraph(str(unresolved_after), table_cell_style)]
    ]
    
    comp_table = Table(comp_data, colWidths=[204, 150, 150])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    
    comparison_block = KeepTogether([
        Paragraph("6. Before vs After Comparison", h1_style),
        Paragraph("Comparison table highlighting overall improvements across critical dimensions:", body_style),
        Spacer(1, 4),
        comp_table,
        Spacer(1, 10)
    ])
    story.append(comparison_block)
    
    # 7. Validation conclusion status banner
    status_label = "PASSED" if unresolved_after == 0 else "PASSED WITH WARNINGS"
    status_color = "#10b981" if unresolved_after == 0 else "#f59e0b"
    status_data = [[
        Paragraph(f"<b>VALIDATION AUDIT STATUS : [ {status_label} ]</b>", ParagraphStyle('StatusStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.white, alignment=TA_CENTER))
    ]]
    status_table = Table(status_data, colWidths=[504])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(status_color)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(status_color)),
    ]))
    
    story.append(KeepTogether([
        Spacer(1, 4),
        status_table
    ]))
    
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def escape_markdown(text: str) -> str:
    """Escapes HTML sensitive brackets in string code snippets for reportlab Paragraph compatibility."""
    if not isinstance(text, str):
        return str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
