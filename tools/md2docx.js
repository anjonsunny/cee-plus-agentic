const fs = require('fs');
const path = require('path');
const D = require('docx');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, PageOrientation, LevelFormat,
  TableOfContents, PageBreak
} = D;

const SRC = process.argv[2];
const OUT = process.argv[3];
const md = fs.readFileSync(SRC, 'utf8').split(/\r?\n/);

// US Letter portrait, 1" margins -> content width 12240 - 2*1440 = 9360 DXA
const CONTENT_W = 9360;

const FONT = 'Calibri';
const MONO = 'Consolas';

function inline(text) {
  // Split on `code`, **bold**, *italic* keeping delimiters
  const runs = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: 20 }));
    const tok = m[0];
    if (tok.startsWith('`')) {
      runs.push(new TextRun({ text: tok.slice(1, -1), font: MONO, size: 18, color: '9A3412' }));
    } else if (tok.startsWith('**')) {
      runs.push(new TextRun({ text: tok.slice(2, -2), font: FONT, size: 20, bold: true }));
    } else {
      runs.push(new TextRun({ text: tok.slice(1, -1), font: FONT, size: 20, italics: true }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), font: FONT, size: 20 }));
  if (runs.length === 0) runs.push(new TextRun({ text: '', font: FONT, size: 20 }));
  return runs;
}

function cellPara(text, opts = {}) {
  return new Paragraph({
    children: inline(text),
    spacing: { before: 40, after: 40 },
    alignment: opts.align || AlignmentType.LEFT,
  });
}

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  return s.split('|').map(c => c.trim());
}

function buildTable(rows) {
  const header = rows[0];
  const body = rows.slice(2); // skip separator
  const n = header.length;

  // Weight columns by content length so wide prose columns get more room
  const weights = new Array(n).fill(0);
  for (const r of [header, ...body]) {
    for (let i = 0; i < n; i++) {
      const len = (r[i] || '').length;
      if (len > weights[i]) weights[i] = len;
    }
  }
  const clamped = weights.map(w => Math.max(11, Math.min(w, 46)));
  const total = clamped.reduce((a, b) => a + b, 0);
  let widths = clamped.map(w => Math.floor(CONTENT_W * w / total));
  const drift = CONTENT_W - widths.reduce((a, b) => a + b, 0);
  widths[widths.length - 1] += drift;

  const mkRow = (cells, isHeader) => new TableRow({
    tableHeader: isHeader,
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: isHeader ? { type: ShadingType.CLEAR, fill: 'EFEDE8' } : undefined,
      margins: { top: 90, bottom: 90, left: 130, right: 130 },
      children: [new Paragraph({
        children: isHeader
          ? [new TextRun({ text: c.replace(/\*\*/g, ''), font: FONT, size: 19, bold: true })]
          : inline(c),
        spacing: { before: 30, after: 30 },
      })],
    })),
  });

  const allRows = [mkRow(header, true), ...body.filter(r => r.some(x => x !== '')).map(r => {
    while (r.length < n) r.push('');
    return mkRow(r.slice(0, n), false);
  })];

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: WidthType.DXA },
    rows: allRows,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: 'C9C6BE' },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: 'C9C6BE' },
      left: { style: BorderStyle.SINGLE, size: 2, color: 'C9C6BE' },
      right: { style: BorderStyle.SINGLE, size: 2, color: 'C9C6BE' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'DEDBD4' },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: 'DEDBD4' },
    },
  });
}

const children = [];

// Title block
const TITLE = process.argv[4] || 'Agentic CEE+';
const SUB = process.argv[5] || '';
children.push(new Paragraph({
  children: [new TextRun({ text: TITLE, font: FONT, size: 56, bold: true })],
  spacing: { after: 80 },
}));
children.push(new Paragraph({
  children: [new TextRun({ text: SUB, font: FONT, size: 24, color: '5F5E5A' })],
  spacing: { after: 60 },
}));
children.push(new Paragraph({
  children: [new TextRun({ text: 'Authored 2026-07-19 · CEE+ · Anjon Basak', font: FONT, size: 20, color: '888780' })],
  spacing: { after: 300 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'C9C6BE', space: 8 } },
}));

children.push(new Paragraph({
  children: [new TextRun({ text: 'Contents', font: FONT, size: 28, bold: true })],
  spacing: { before: 200, after: 120 },
}));
children.push(new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-3' }));
children.push(new Paragraph({ children: [new PageBreak()] }));

let i = 0;
let skippedTitle = false;

while (i < md.length) {
  const line = md[i];

  // Table block
  if (/^\s*\|/.test(line) && i + 1 < md.length && /^\s*\|[\s:|-]+\|?\s*$/.test(md[i + 1])) {
    const rows = [];
    while (i < md.length && /^\s*\|/.test(md[i])) { rows.push(splitRow(md[i])); i++; }
    children.push(buildTable(rows));
    children.push(new Paragraph({ text: '', spacing: { after: 140 } }));
    continue;
  }

  // Headings
  let m;
  if ((m = line.match(/^###\s+(.*)$/))) {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: m[1].replace(/\*\*/g, ''), font: FONT, size: 24, bold: true, color: '1F1F1D' })],
      spacing: { before: 260, after: 100 },
    }));
    i++; continue;
  }
  if ((m = line.match(/^##\s+(.*)$/))) {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: m[1].replace(/\*\*/g, ''), font: FONT, size: 30, bold: true, color: '1F1F1D' })],
      spacing: { before: 340, after: 130 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'DEDBD4', space: 6 } },
    }));
    i++; continue;
  }
  if ((m = line.match(/^#\s+(.*)$/))) {
    if (!skippedTitle) { skippedTitle = true; i++; continue; }
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: m[1], font: FONT, size: 36, bold: true })],
      spacing: { before: 360, after: 150 },
    }));
    i++; continue;
  }

  // Horizontal rule -> spacing + rule
  if (/^---+\s*$/.test(line)) {
    children.push(new Paragraph({
      text: '',
      spacing: { before: 120, after: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'DEDBD4', space: 4 } },
    }));
    i++; continue;
  }

  // Blockquote (callout)
  if ((m = line.match(/^>\s?(.*)$/))) {
    const buf = [];
    while (i < md.length && /^>/.test(md[i])) { buf.push(md[i].replace(/^>\s?/, '')); i++; }
    const text = buf.filter(x => x.trim() !== '').join(' ');
    children.push(new Paragraph({
      children: inline(text),
      spacing: { before: 120, after: 140, line: 300 },
      indent: { left: 240, right: 120 },
      shading: { type: ShadingType.CLEAR, fill: 'FBF3E4' },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'BA7517', space: 10 } },
    }));
    continue;
  }

  // Bullets (support one nesting level)
  if ((m = line.match(/^(\s*)[-*]\s+(.*)$/))) {
    const indentLvl = Math.floor(m[1].length / 2);
    children.push(new Paragraph({
      children: inline(m[2]),
      bullet: { level: Math.min(indentLvl, 1) },
      spacing: { before: 50, after: 50, line: 288 },
      indent: { left: 360 + indentLvl * 340, hanging: 230 },
    }));
    i++; continue;
  }

  // Numbered list
  if ((m = line.match(/^(\s*)(\d+)\.\s+(.*)$/))) {
    children.push(new Paragraph({
      children: [new TextRun({ text: m[2] + '.  ', font: FONT, size: 20, bold: true }), ...inline(m[3])],
      indent: { left: 360 + (Math.floor(m[1].length / 2) * 300), hanging: 300 },
      spacing: { before: 50, after: 50, line: 288 },
    }));
    i++; continue;
  }

  // Blank
  if (line.trim() === '') { i++; continue; }

  // Paragraph: join continuation lines
  const buf = [line];
  i++;
  while (i < md.length && md[i].trim() !== '' && !/^[#>\-*|]/.test(md[i]) && !/^\d+\.\s/.test(md[i]) && !/^\s*\|/.test(md[i])) {
    buf.push(md[i]); i++;
  }
  children.push(new Paragraph({
    children: inline(buf.join(' ')),
    spacing: { before: 60, after: 110, line: 300 },
  }));
}

const doc = new Document({
  creator: 'Anjon Basak',
  title: TITLE,
  description: 'Agentic rebuild plan for CEE+: 25 stages, loops, evaluation strategy, build order',
  styles: {
    default: {
      document: { run: { font: FONT, size: 20 }, paragraph: { spacing: { line: 300 } } },
      heading1: { run: { font: FONT, size: 36, bold: true, color: '1F1F1D' } },
      heading2: { run: { font: FONT, size: 30, bold: true, color: '1F1F1D' } },
      heading3: { run: { font: FONT, size: 24, bold: true, color: '1F1F1D' } },
    },
  },
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 220 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '◦', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 220 } } } },
      ],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log('wrote', OUT, (buf.length / 1024).toFixed(0) + 'KB');
});
