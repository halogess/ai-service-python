# Alignment Visualization

Fitur untuk menggambar hasil alignment pada gambar halaman PDF.

## Cara Kerja

1. **Ekstraksi PDF**: PDF dikonversi menjadi gambar (300 DPI) dan diekstrak kontennya
2. **Alignment**: Konten PDF di-align dengan DokumenElemen dari database
3. **Visualisasi**: Hasil alignment digambar pada gambar dengan bounding box berwarna:
   - **Hijau**: Item yang berhasil di-align
   - **Merah**: Item yang tidak ter-align
   - **Orange**: Item header/footer

## Struktur Folder

```
storage/
├── dokumen/
│   ├── pdf/
│   │   └── document.pdf
│   ├── alignment/          # ← Folder baru untuk visualisasi
│   │   ├── page_001.png
│   │   ├── page_002.png
│   │   └── ...
│   └── extraction/
│       └── document/
│           ├── page_001.png
│           ├── extraction_results.json
│           └── alignment_results.json
```

## Penggunaan

Visualisasi alignment akan otomatis dibuat saat visual worker memproses dokumen:

```python
# Visual worker akan:
# 1. Ekstrak PDF dan buat gambar
# 2. Jalankan alignment
# 3. Gambar hasil alignment pada gambar
# 4. Simpan ke folder /alignment
```

## File yang Dihasilkan

- `page_001.png`, `page_002.png`, ... : Gambar dengan alignment yang digambar
- Lokasi: `{pdf_directory}/alignment/`

## Konfigurasi

Warna bounding box dapat diubah di `utils/alignment_visualizer.py`:

```python
self.colors = {
    'aligned': '#00FF00',      # Hijau
    'unaligned': '#FF0000',    # Merah
    'header_footer': '#FFA500' # Orange
}
```

## Dependencies

- Pillow (PIL): Untuk menggambar pada gambar
- PyMuPDF (fitz): Untuk konversi PDF ke gambar
