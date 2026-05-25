#!/usr/bin/env python3
"""
Academic PDF Compiler using ReportLab
Copyright (c) 2026 Xavier Callens / Socrate AI Lab
All rights reserved.

Compiles Markdown drafts to high-fidelity, publication-grade academic PDFs.
"""
import re
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

def clean_md_inline(text):
    """Convert Markdown inline styles to ReportLab HTML-like tags."""
    # 1. Extract inline code blocks to avoid them being styled as bold/italic
    code_blocks = []
    def placeholder_code(match):
        code_blocks.append(match.group(1))
        return f"__CODE_PLACEHOLDER_{len(code_blocks)-1}__"
    
    # Temporarily substitute inline code
    text = re.sub(r'`(.*?)`', placeholder_code, text)
    
    # 2. Run standard bold/italic markdown replacements
    # Convert bold-italic (***text*** or ___text___)
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'___(.*?)___', r'<b><i>\1</i></b>', text)
    # Convert bold (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    # Convert italic (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    
    # 3. Restore inline code and wrap them in font tags (escaping HTML entities)
    for idx, code in enumerate(code_blocks):
        escaped_code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        font_tag = f'<font name="Courier" size="9" color="#C7254E" backColor="#F9F2F4">{escaped_code}</font>'
        text = text.replace(f"__CODE_PLACEHOLDER_{idx}__", font_tag)
        
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text)
    return text.strip()

def parse_markdown_to_story(md_path, styles):
    story = []
    
    # Custom Styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A2B4C'),
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#3C4E6E'),
        alignment=TA_CENTER,
        spaceAfter=4
    )
    
    affiliation_style = ParagraphStyle(
        'PaperAffiliation',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#6C7E9E'),
        alignment=TA_CENTER,
        spaceAfter=16
    )
    
    h2_style = ParagraphStyle(
        'PaperH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1A2B4C'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'PaperH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#3C4E6E'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    abstract_body_style = ParagraphStyle(
        'PaperAbstractBody',
        parent=body_style,
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#333333'),
        leftIndent=24,
        rightIndent=24,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    abstract_header_style = ParagraphStyle(
        'PaperAbstractHeader',
        parent=h2_style,
        fontName='Helvetica-Bold',
        fontSize=11,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    quote_style = ParagraphStyle(
        'PaperQuote',
        parent=body_style,
        fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#4A5A7A'),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=8
    )
    
    code_block_style = ParagraphStyle(
        'PaperCodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#002B36'),
        leftIndent=16,
        rightIndent=16,
        spaceAfter=8
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#333333')
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=table_cell_style,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_lines = []
    
    in_table = False
    table_rows = []
    
    is_abstract = False
    
    title_processed = False
    author_processed = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Image check e.g. ![caption](image_path)
        if stripped.startswith('![') and stripped.endswith(')'):
            match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
            if match:
                caption = match.group(1)
                img_name = match.group(2)
                
                md_dir = os.path.dirname(os.path.abspath(md_path))
                resolved_img_path = os.path.join(md_dir, img_name)
                if not os.path.exists(resolved_img_path):
                    resolved_img_path = os.path.join(os.getcwd(), img_name)
                
                try:
                    from reportlab.platypus import Image as RLImage
                    rl_img = RLImage(resolved_img_path, width=320, height=320)
                    rl_img.hAlign = 'CENTER'
                    story.append(rl_img)
                    story.append(Spacer(1, 6))
                    
                    caption_style = ParagraphStyle(
                        'ImageCaption',
                        parent=styles['Normal'],
                        fontName='Helvetica-Bold',
                        fontSize=8,
                        leading=10,
                        textColor=colors.HexColor('#555555'),
                        alignment=TA_CENTER,
                        spaceAfter=12
                    )
                    story.append(Paragraph(f"Figure: {caption}", caption_style))
                except Exception as e:
                    story.append(Paragraph(f"[Image could not be loaded: {e}]", body_style))
                i += 1
                continue
        
        # Monospace code block check
        if line.strip().startswith('```'):
            if in_code_block:
                in_code_block = False
                # Compile code block
                code_text = "<br/>".join(code_lines)
                
                table_data = [[Paragraph(code_text, code_block_style)]]
                cb_table = Table(table_data, colWidths=[480])
                cb_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
                    ('PADDING', (0,0), (-1,-1), 8),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E9ECEF')),
                ]))
                story.append(cb_table)
                story.append(Spacer(1, 8))
                code_lines = []
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            # Escape HTML special characters first to avoid double-escaping
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Replace spaces and tabs with non-breaking spaces
            processed_line = escaped_line.replace(' ', '&nbsp;').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').rstrip('\r\n')
            code_lines.append(processed_line)
            i += 1
            continue
            
        # Table parsing
        if line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            
            # Skip separator line e.g., |---|---|
            if '---' in line:
                i += 1
                continue
                
            # Process table row
            cells = [clean_md_inline(c.strip()) for c in line.split('|')[1:-1]]
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            # Compile Table
            in_table = False
            if len(table_rows) > 0:
                header = table_rows[0]
                body = table_rows[1:]
                
                formatted_data = []
                # Header row
                formatted_data.append([Paragraph(cell, table_header_style) for cell in header])
                # Body rows
                for r in body:
                    formatted_data.append([Paragraph(cell, table_cell_style) for cell in r])
                
                col_widths = [500.0 / len(header)] * len(header)
                t = Table(formatted_data, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A2B4C')),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('PADDING', (0,0), (-1,-1), 5),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9FA')]),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E9ECEF')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CED4DA')),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
            table_rows = []
            
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
            
        # Headers
        if stripped.startswith('# '):
            title = clean_md_inline(stripped[2:])
            story.append(Paragraph(title, title_style))
            title_processed = True
            
            # Look ahead for authors
            if i + 1 < len(lines) and lines[i+1].strip().startswith('**Authors**'):
                author_line = lines[i+1].strip()
                authors = clean_md_inline(author_line.replace('**Authors**:', '').replace('**Author**:', '').strip())
                story.append(Paragraph(authors, author_style))
                author_processed = True
                i += 2
                
                # Check for affiliation/organization
                if i < len(lines) and 'Lab' in lines[i]:
                    affil = clean_md_inline(lines[i])
                    story.append(Paragraph(affil, affiliation_style))
                    i += 1
                continue
                
        elif stripped.startswith('## '):
            hdr = clean_md_inline(stripped[3:])
            if hdr.upper() == 'ABSTRACT':
                is_abstract = True
                story.append(Paragraph("ABSTRACT", abstract_header_style))
            else:
                is_abstract = False
                story.append(Paragraph(hdr, h2_style))
            i += 1
            continue
            
        elif stripped.startswith('### '):
            hdr = clean_md_inline(stripped[4:])
            story.append(Paragraph(hdr, h3_style))
            i += 1
            continue
            
        # Blockquote/Quotes
        if stripped.startswith('> '):
            quote = clean_md_inline(stripped[2:])
            # Styled box for quote
            table_data = [[Paragraph(quote, quote_style)]]
            q_table = Table(table_data, colWidths=[480])
            q_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EDF2F7')),
                ('PADDING', (0,0), (-1,-1), 8),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('LINELEFT', (0,0), (-1,-1), 3.0, colors.HexColor('#3182CE')),
            ]))
            story.append(q_table)
            story.append(Spacer(1, 8))
            i += 1
            continue
            
        # Bullet list items
        if stripped.startswith('- '):
            bullet_text = clean_md_inline(stripped[2:])
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{bullet_text}", body_style))
            i += 1
            continue
        elif stripped.startswith('* '):
            bullet_text = clean_md_inline(stripped[2:])
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{bullet_text}", body_style))
            i += 1
            continue
        elif stripped.startswith('1. '):
            num_text = clean_md_inline(stripped[3:])
            story.append(Paragraph(f"1.&nbsp;&nbsp;{num_text}", body_style))
            i += 1
            continue
            
        # Standard Paragraph
        text = clean_md_inline(stripped)
        if text:
            if is_abstract:
                story.append(Paragraph(text, abstract_body_style))
            else:
                story.append(Paragraph(text, body_style))
                
        i += 1
        
    return story

def compile_pdf(md_path, pdf_path):
    print(f"Compiling: {md_path} -> {pdf_path}")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    story = parse_markdown_to_story(md_path, styles)
    
    # Custom Page numbering
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8.5)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(54, 30, "RunuX-AI Scientific Publication | Socrate AI Lab")
        canvas.drawRightString(doc.pagesize[0]-54, 30, f"Page {doc.page}")
        canvas.restoreState()
        
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"✓ PDF successfully generated: {pdf_path}\n")

def main():
    # 1. Compile RunuX-AI Systolic Runtime Article
    src_dir = os.path.dirname(os.path.abspath(__file__))
    scientific_md = os.path.join(src_dir, "scientific_article.md")
    scientific_pdf = os.path.join(src_dir, "RunuX_AI_Systolic_Runtime_Article.pdf")
    compile_pdf(scientific_md, scientific_pdf)
    
    # 2. Compile Quantum LTN Simulator Draft
    quantum_dir = "/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/scripts/quantum_ltn"
    quantum_md = os.path.join(quantum_dir, "PAPER_DRAFT.md")
    quantum_pdf = os.path.join(quantum_dir, "Dynamics_of_Disordered_Quantum_Systems_via_Telemetry_Guided_3D_Logic_Tensor_Networks_in_Safe_Systems_Runtimes.pdf")
    if os.path.exists(quantum_md):
        compile_pdf(quantum_md, quantum_pdf)
    else:
        print(f"⚠ Quantum LTN draft not found at {quantum_md}")

    # 3. Compile Lean 4 Formal Proof Paper
    proof_md = os.path.join(src_dir, "spec_proof_paper.md")
    proof_pdf = os.path.join(src_dir, "RunuX_Lean4_Formal_Proof_Paper.pdf")
    if os.path.exists(proof_md):
        compile_pdf(proof_md, proof_pdf)
    else:
        print(f"⚠ Lean 4 formal proof paper not found at {proof_md}")

    # 4. Compile Auto-Research Agent Architecture Proposal
    proposal_md = "/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/docs/AUTO_RESEARCH_AGENT_ARCHITECTURE.md"
    proposal_pdf = "/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/docs/RunuX_Auto_Research_Agent_Architecture.pdf"
    if os.path.exists(proposal_md):
        compile_pdf(proposal_md, proposal_pdf)
    else:
        print(f"⚠ Auto-Research Agent Architecture Proposal not found at {proposal_md}")

if __name__ == "__main__":
    main()
