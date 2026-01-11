# Difflib Alignment - Modular Structure

Dekomposisi dari `difflib_alignment.py` (2747 baris) menjadi modul-modul yang lebih maintainable.

## Struktur Modul

### Core Configuration
- **`config.py`** - Konstanta dan regex patterns

### Text Processing
- **`text_utils.py`** - Normalisasi dan tokenisasi teks
- **`formula_utils.py`** - Normalisasi khusus formula matematika

### DOCX Extraction
- **`docx_extractor.py`** - Main extractor dengan routing logic
- **`docx_text_extractor.py`** - Ekstraksi teks rekursif
- **`docx_table_extractor.py`** - Ekstraksi table cells
- **`docx_shape_extractor.py`** - Ekstraksi shapes
- **`docx_tokenizer.py`** - Build DOCX token stream

### PDF Extraction
- **`pdf_extractor.py`** - Ekstraksi token dan table dari PDF
- **`pdf_tokenizer.py`** - Build PDF token stream
- **`image_extractor.py`** - Ekstraksi raster & vector images

### Alignment Core
- **`alignment_core.py`** - Global difflib alignment
- **`table_matcher.py`** - Match DOCX cell ke PDF cell
- **`bbox_utils.py`** - Merge dan manipulasi bounding boxes

### Element Builders
- **`element_builder.py`** - Build metadata dan orchestrate builders
- **`shape_element_builder.py`** - Build shape elements
- **`table_builder.py`** - Build table containers
- **`table_cell_processor.py`** - Process individual table cells
- **`non_table_builder.py`** - Build paragraphs dan non-table elements

### Image Alignment
- **`image_aligner.py`** - Align images dalam table cells
- **`image_alignment_workflow.py`** - Workflow untuk standalone images

### Main & Result
- **`main.py`** - Orchestrate semua modul
- **`result_builder.py`** - Build final result dictionary

## Usage

```python
from difflib_alignment import align_document

result = align_document(pdf_path, elements, log_file=None)
```

## Benefits

1. **Separation of Concerns** - Setiap modul punya tanggung jawab spesifik
2. **Testability** - Mudah untuk unit testing
3. **Reusability** - Modul dapat digunakan independen
4. **Maintainability** - Lebih mudah debug dan update
5. **Readability** - Struktur yang jelas dan logis
