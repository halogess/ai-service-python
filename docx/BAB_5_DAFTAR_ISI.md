# BAB 5 - ANALISA VISUAL DOKUMEN

## Daftar Isi Lengkap dengan Penjelasan

---

## 5.1 Pendahuluan dan Arsitektur

### 5.1.1 Latar Belakang Analisa Visual Dokumen

#### 5.1.1.1 Kebutuhan Analisa Visual dalam Validasi Tugas Akhir
Sistem validasi tugas akhir memerlukan kemampuan untuk memverifikasi format dokumen tidak hanya dari struktur OpenXML, tetapi juga dari tampilan visual PDF yang dihasilkan. Hal ini penting karena beberapa elemen format hanya dapat divalidasi melalui representasi visual.

#### 5.1.1.2 Keterbatasan Ekstraksi OpenXML Murni
Ekstraksi OpenXML dari file DOCX memberikan informasi struktural yang kaya, namun tidak memberikan informasi posisi visual elemen di halaman. Koordinat bounding box hanya tersedia pada PDF yang dirender.

#### 5.1.1.3 Peran PDF sebagai Sumber Validasi Sekunder
PDF menyediakan representasi visual final dokumen dengan koordinat tepat setiap elemen. Service ini mengekstrak data visual dari PDF menggunakan PyMuPDF (fitz) untuk mendapatkan posisi dan label setiap elemen.

#### 5.1.1.4 Tujuan Integrasi Dua Sumber Data
Dengan menggabungkan data OpenXML (struktur dan format) dengan data PDF (posisi visual), sistem dapat melakukan validasi yang lebih komprehensif seperti pengecekan margin, posisi header/footer, dan layout halaman.

### 5.1.2 Arsitektur Sistem Analisa Visual

#### 5.1.2.1 Komponen Utama Sistem
Sistem terdiri dari: `visual_worker.py` (entry point), `MergingExtractionService` (orkestrator), `PDFExtractor` (ekstraksi PDF), `AlignmentService` (penyelarasan), `DoclingService` (klasifikasi AI), `DoclingFusionService` (penggabungan), dan `VisualizationService` (output visual).

#### 5.1.2.2 Alur Data dari Input hingga Output
Alur dimulai dari antrian task → ekstraksi PDF → alignment dengan OpenXML → klasifikasi Docling → fusion hasil → pemberian label struktural → penyimpanan ke tabel `dokumen_elemen_visual`.

#### 5.1.2.3 Hubungan dengan Service Ekstraksi (Project Lain)
Service ini membaca data `DokumenElemen` yang telah diekstrak oleh service OpenXML dari project .NET. Data yang dibaca termasuk `delemen_json_tree`, `delemen_text`, dan `delemen_sequence`.

#### 5.1.2.4 Hubungan dengan Service Validasi (Project Lain)
Setelah labeling selesai, status antrian diubah ke `antrian_validation_status = 'in_queue'` agar dapat diproses oleh validation worker di project yang sama dengan service ekstraksi.

### 5.1.3 Teknologi dan Library yang Digunakan

#### 5.1.3.1 Python sebagai Bahasa Pemrograman
Python dipilih karena ekosistem AI/ML yang kuat dan ketersediaan library seperti Docling untuk document understanding.

#### 5.1.3.2 FastAPI sebagai Framework Web Service
FastAPI digunakan untuk menyediakan API endpoint dengan dokumentasi otomatis dan performa tinggi berbasis async.

#### 5.1.3.3 SQLAlchemy untuk Object-Relational Mapping
SQLAlchemy ORM digunakan untuk interaksi dengan database PostgreSQL, dengan model-model seperti `Dokumen`, `Antrian`, `DokumenElemen`, dan `DokumenElemenVisual`.

#### 5.1.3.4 PyMuPDF (fitz) untuk Manipulasi PDF
PyMuPDF (import fitz) menyediakan kemampuan high-performance untuk membaca teks, karakter, gambar, dan drawing dari PDF dengan koordinat bounding box.

#### 5.1.3.5 IBM Docling untuk Klasifikasi Dokumen
Docling adalah library dari IBM untuk document understanding yang dapat mengklasifikasikan elemen dokumen ke label seperti text, section_header, table, picture, caption, dll.

### 5.1.4 Struktur Direktori Project

#### 5.1.4.1 Organisasi Folder src/services
Berisi service utama: `alignment_service.py`, `docling_service.py`, `docling_fusion_service.py`, `merging_extraction_service.py`, `pdf_extraction_service.py`, `visualization_service.py`, dan subfolder `alignment/` dengan 4 mixin.

#### 5.1.4.2 Organisasi Folder src/workers
Berisi `visual_worker.py` sebagai entry point polling antrian dan `merging_worker.py`.

#### 5.1.4.3 Organisasi Folder src/models
Berisi model SQLAlchemy: `antrian.py`, `dokumen.py`, `dokumen_elemen.py`, `dokumen_elemen_visual.py`, `dokumen_section.py`, `dokumen_part.py`, `dokumen_note.py`.

#### 5.1.4.4 Organisasi Folder src/utils
Berisi utility functions: `char_grouping.py` (algoritma pengelompokan karakter), `alignment_core.py`, `alignment_visualizer.py`.

---

## 5.2 Visual Worker (Pemrosesan Antrian)

### 5.2.1 Konsep Queue-Based Processing

#### 5.2.1.1 Arsitektur Antrian untuk Pemrosesan Asinkron
Sistem menggunakan tabel `antrian` dengan kolom `antrian_labeling_status` untuk mengelola task secara asinkron. Worker melakukan polling secara periodik untuk mengambil task yang berstatus 'in_queue'.

#### 5.2.1.2 Model Status Task (in_queue, processing, completed, failed)
Di model `Antrian`, field status menggunakan Enum dengan nilai: 'in_queue' (menunggu), 'processing' (sedang diproses), 'completed' (selesai), 'failed' (gagal dengan pesan error di `antrian_error_message`).

#### 5.2.1.3 Polling Mechanism dengan Interval Waktu
Fungsi `run_visual_worker(check_interval=5)` menjalankan loop tak terbatas dengan `time.sleep(check_interval)` di antara setiap pengecekan antrian.

#### 5.2.1.4 Keuntungan Pemrosesan Berbasis Antrian
Antrian memungkinkan pemrosesan yang scalable, retry pada kegagalan, dan decoupling antara upload dokumen dengan pemrosesan.

### 5.2.2 Implementasi File visual_worker.py

#### 5.2.2.1 Struktur dan Import Dependencies
File mengimport: `SessionLocal`, `engine` dari database, model `Base`, `Dokumen`, service `AntrianService`, `MergingExtractionService`, dan utility `convert_pdf_to_images`.

#### 5.2.2.2 Konfigurasi Logging ke File dan Console
Logging dikonfigurasi dengan `logging.basicConfig()` yang menulis ke file `logs/visual_worker.log` dan console dengan format timestamp-level-message.

#### 5.2.2.3 Inisialisasi Database dengan SQLAlchemy Engine
Perintah `Base.metadata.create_all(bind=engine)` memastikan semua tabel model sudah ada di database sebelum worker mulai.

#### 5.2.2.4 Entry Point dengan Fungsi __main__
Blok `if __name__ == "__main__": run_visual_worker(check_interval=5)` memastikan worker berjalan ketika file dieksekusi langsung.

### 5.2.3 Fungsi process_visual_task()

#### 5.2.3.1 Pengambilan Session Database
Fungsi membuat session baru dengan `db = SessionLocal()` dan memastikan ditutup di `finally` block.

#### 5.2.3.2 Interaksi dengan AntrianService untuk Mendapat Task
`AntrianService(db).get_next_labeling_task()` mengquery antrian untuk task dengan `antrian_labeling_status == 'in_queue'` dan mengurutkan berdasarkan `antrian_created_at`.

#### 5.2.3.3 Update Status ke 'processing'
Sebelum memproses, status diupdate dengan `antrian_service.update_labeling_status(task, 'processing')` untuk mencegah worker lain mengambil task yang sama.

#### 5.2.3.4 Pengambilan PDF Path dan Output Directory
`antrian_service.get_full_pdf_path(task)` mengembalikan path absolut ke PDF dengan menggabungkan `STORAGE_BASE` dengan `dokumen.dokumen_pdf_path`.

### 5.2.4 Konversi PDF ke Images

#### 5.2.4.1 Pemanggilan convert_pdf_to_images()
Fungsi dari `pdf_image_service.py` mengkonversi setiap halaman PDF menjadi gambar untuk keperluan preview dan debugging.

#### 5.2.4.2 Update dokumen_images_path di Database
Setelah konversi, path images disimpan ke `dokumen.dokumen_images_path` dengan format relatif `/dokumen/{nrp}/{dokumen_id}/images`.

#### 5.2.4.3 Struktur Direktori Images per Dokumen
Images disimpan dalam folder terpisah per dokumen untuk memudahkan pengelolaan dan akses melalui API.

### 5.2.5 Delegasi ke MergingExtractionService

#### 5.2.5.1 Inisialisasi MergingExtractionService
Service diinisialisasi dengan `MergingExtractionService()` yang di dalamnya menginisialisasi semua sub-service (AlignmentService, DoclingService, dll).

#### 5.2.5.2 Parameter process_document (doc_id, visualizations, save_to_db)
Fungsi dipanggil dengan `process_document(doc_id=task.dokumen_id, generate_visualizations=True, save_to_db=True, output_dir=classification_dir)`.

#### 5.2.5.3 Penentuan classification_dir untuk Output
Output visualisasi disimpan ke folder `{pdf_dir}/classification/` yang dibuat dengan `os.makedirs(classification_dir, exist_ok=True)`.

#### 5.2.5.4 Transisi Status ke Validation Queue setelah Sukses
Jika berhasil, `antrian_service.update_validation_status(task, 'in_queue')` dipanggil untuk memindahkan task ke antrian validasi.

### 5.2.6 Error Handling dan Recovery

#### 5.2.6.1 Exception Handling dengan try-except
Seluruh proses dibungkus dalam try-except untuk menangkap semua error yang mungkin terjadi.

#### 5.2.6.2 Database Rollback pada Kegagalan
Jika error terjadi, `db.rollback()` dipanggil untuk membatalkan perubahan yang belum di-commit.

#### 5.2.6.3 Logging Error dengan Stack Trace
Error di-log dengan `logger.error(f"...", exc_info=True)` untuk menyertakan stack trace lengkap.

#### 5.2.6.4 Update Status ke 'failed' dengan Pesan Error
Status task diupdate ke 'failed' dengan pesan error: `antrian_service.update_labeling_status(task, 'failed', str(e))`.

### 5.2.7 Fungsi run_visual_worker()

#### 5.2.7.1 Main Loop dengan While True
Worker berjalan dalam infinite loop `while True:` yang terus memanggil `process_visual_task()`.

#### 5.2.7.2 Check Interval untuk Polling
Setelah setiap iterasi, `time.sleep(check_interval)` memberikan jeda sebelum pengecekan berikutnya.

#### 5.2.7.3 Graceful Shutdown dengan KeyboardInterrupt
Ctrl+C ditangani dengan `except KeyboardInterrupt:` untuk shutdown yang bersih dengan logging.

#### 5.2.7.4 Pembuatan Direktori Logs
Perintah `os.makedirs('logs', exist_ok=True)` memastikan folder logs sudah ada.

---

## 5.3 Ekstraksi PDF dengan PyMuPDF

### 5.3.1 Pengenalan Library PyMuPDF (fitz)

#### 5.3.1.1 Sejarah dan Perkembangan PyMuPDF
PyMuPDF adalah binding Python untuk library MuPDF yang dikembangkan oleh Artifex Software, menyediakan akses cepat ke konten PDF.

#### 5.3.1.2 Kemampuan Ekstraksi Multi-Format
PyMuPDF dapat mengekstrak teks, gambar, metadata, annotations, dan drawing paths dari berbagai format termasuk PDF, XPS, EPUB.

#### 5.3.1.3 Keunggulan Performa dibanding Library Lain
Benchmarks menunjukkan PyMuPDF lebih cepat dari PyPDF2 dan pdfminer untuk ekstraksi teks dengan memory footprint lebih kecil.

#### 5.3.1.4 Dokumentasi dan Komunitas PyMuPDF
Library memiliki dokumentasi lengkap di pymupdf.readthedocs.io dan komunitas aktif di GitHub.

### 5.3.2 Implementasi Class PDFExtractor

#### 5.3.2.1 Inisialisasi dengan Path PDF
Constructor `__init__(self, pdf_path: str)` menyimpan path dan menginisialisasi `self.doc = None`.

#### 5.3.2.2 Context Manager (__enter__, __exit__)
Class mengimplementasikan context manager protocol untuk penggunaan dengan `with PDFExtractor(path) as extractor:`.

#### 5.3.2.3 Metode open() dan close()
`open()` membuka dokumen dengan `fitz.open(self.pdf_path)`, `close()` menutup dengan `self.doc.close()`.

#### 5.3.2.4 Property page_count dan get_page()
`page_count` mengembalikan jumlah halaman, `get_page(page_num)` mengembalikan objek halaman dengan validasi index.

### 5.3.3 Metode extract_merging_data()

#### 5.3.3.1 Ekstraksi Text Dictionary dengan rawdict
Menggunakan `page.get_text("rawdict")` untuk mendapatkan struktur lengkap teks termasuk karakter, font, dan posisi.

#### 5.3.3.2 Penanganan Sanitasi Karakter Non-Serializable
Fungsi `sanitize(obj)` memproses dictionary secara rekursif untuk menangani karakter yang tidak bisa di-serialize ke JSON.

#### 5.3.3.3 Pengumpulan Character Groups
Memanggil `find_overlapping_groups()` dari `char_grouping.py` kemudian diurutkan dengan `sort_groups_reading_order()`.

#### 5.3.3.4 Pengumpulan Gambar dan Shapes
Menggunakan `page.get_images()` untuk gambar dan `page.get_drawings()` untuk shapes dengan koordinat bbox.

### 5.3.4 Character Grouping dengan DFS

#### 5.3.4.1 Konsep Overlapping Character pada PDF
Karakter yang secara visual overlapping (misal subscript, superscript, atau fraction) perlu dikelompokkan bersama untuk mempertahankan urutan baca.

#### 5.3.4.2 Toleransi X (X_TOLERANCE = 3 points)
Dua karakter dianggap overlap secara X jika koordinat x0 karakter kedua berada dalam range x0-x1 karakter pertama ± 3 points.

#### 5.3.4.3 Toleransi Y (Y_OVERLAP_MIN_RATIO = 0.3)
Dua karakter dianggap overlap secara Y jika area overlap vertikal ≥ 30% dari tinggi karakter yang lebih pendek.

#### 5.3.4.4 Algoritma find_overlapping_groups
Menggunakan Depth-First Search (DFS) untuk menemukan semua karakter yang saling overlapping dan menggabungkannya menjadi satu group dengan merged_bbox.

#### 5.3.4.5 Pengurutan Reading Order
`sort_groups_reading_order()` mengurutkan groups berdasarkan line (Y overlap) lalu left-to-right dalam line yang sama.

### 5.3.5 Deteksi Segmen Garis

#### 5.3.5.1 Fungsi collect_horizontal_segments()
Mengiterasi `page.get_drawings()` untuk mengumpulkan garis horizontal (y0 ≈ y1) dengan koordinat (y, x0, x1).

#### 5.3.5.2 Fungsi collect_vertical_segments()
Mengumpulkan garis vertikal (x0 ≈ x1) yang berguna untuk deteksi batas kolom tabel.

#### 5.3.5.3 Penggabungan Segmen dengan merge_by_y_and_x()
Menggabungkan segmen horizontal yang berada pada Y yang sama (dalam toleransi) dan X yang bersebelahan menjadi satu segmen panjang.

#### 5.3.5.4 Kegunaan Segmen untuk Deteksi Tabel
Segmen horizontal yang berulang pada Y tertentu mengindikasikan garis tabel, digunakan untuk menentukan area clip tabel.

### 5.3.6 Deteksi Tabel dari Horizontal Lines

#### 5.3.6.1 Fungsi guess_table_clips_from_hlines()
Mendeteksi area tabel berdasarkan pola garis horizontal berulang dengan minimal 2 garis (`min_rules=2`).

#### 5.3.6.2 Penentuan Area Clip untuk Tabel
Clip adalah rectangular area yang mencakup semua garis horizontal yang membentuk tabel, dengan padding vertikal.

#### 5.3.6.3 Validasi dengan Character Groups
Clip divalidasi dengan mengecek apakah ada character groups di dalam area tersebut dan apakah layout konsisten dengan tabel.

#### 5.3.6.4 Deteksi has_horizontal_outlier()
Fungsi helper untuk mengecek apakah ada karakter yang melampaui batas tabel secara horizontal, yang mengindikasikan false positive.

### 5.3.7 Deteksi Sel dalam Tabel

#### 5.3.7.1 Fungsi detect_cells_from_segments()
Mendeteksi sel-sel tabel berdasarkan interseksi segmen horizontal dan vertikal dalam area clip.

#### 5.3.7.2 Penggabungan Posisi dengan merge_close_positions()
Posisi Y yang berdekatan (dalam toleransi) digabungkan untuk menangani variasi minor pada garis tabel.

#### 5.3.7.3 Deteksi Batas Kolom dari Teks
Jika tidak ada garis vertikal, batas kolom diestimasi dari gap horizontal antar character groups dengan `detect_column_boundaries_from_text()`.

#### 5.3.7.4 Pembentukan Grid Sel
Hasil adalah grid 2D dari sel dengan koordinat bbox masing-masing untuk ekstraksi konten per sel.

### 5.3.8 Ekstraksi Gambar dari PDF

#### 5.3.8.1 Identifikasi Objek Gambar di Halaman
`page.get_images()` mengembalikan list tuple dengan informasi gambar termasuk xref untuk ekstraksi.

#### 5.3.8.2 Ekstraksi Bounding Box Gambar
Transformasi matrix gambar digunakan untuk menghitung koordinat bbox di halaman.

#### 5.3.8.3 Penanganan Overlap Gambar dengan Teks
Fungsi `group_overlaps_image()` dan `simple_bbox_overlap()` mengecek apakah text group berada di atas gambar.

### 5.3.9 Pengurutan dan Pengelompokan Output

#### 5.3.9.1 Sorting berdasarkan Posisi Y dan X
Output diurutkan secara top-to-bottom kemudian left-to-right dengan `compare_groups()`.

#### 5.3.9.2 Penanganan Elemen Multi-Baris
Elemen yang tinggi (misal gambar) tidak memecah urutan baca elemen teks yang berada di baris yang sama.

#### 5.3.9.3 Struktur Output Dictionary
Output berisi: `char_groups`, `basic_tables`, `hline_tables`, `shapes`, `page_images` dengan masing-masing termasuk `bbox`.

---

## 5.4 Alignment (Penyelarasan OpenXML-PDF)

### 5.4.1 Konsep dan Tujuan Alignment

#### 5.4.1.1 Permasalahan Perbedaan Representasi OpenXML dan PDF
OpenXML menyimpan dokumen dalam struktur hierarki (paragraf → run → text), sedangkan PDF menyimpan sebagai karakter individual dengan koordinat. Alignment menjembatani kedua representasi ini.

#### 5.4.1.2 Kebutuhan Menghubungkan Elemen dari Dua Sumber
Untuk validasi, kita perlu tahu elemen mana dari OpenXML yang berkorespendensi dengan bounding box mana di PDF.

#### 5.4.1.3 Pendekatan Character-Level Matching
Algoritma menggunakan difflib.SequenceMatcher untuk mencocokkan string teks dari PDF dengan teks dari OpenXML element.

#### 5.4.1.4 Arsitektur Mixin-Based untuk Modularitas
AlignmentService menggunakan 4 mixin class untuk memisahkan concerns: Preprocess, OpenXml, Matching, dan Postprocess.

### 5.4.2 AlignmentService sebagai Koordinator

#### 5.4.2.1 Struktur Class AlignmentService
Class mewarisi dari 4 mixin dan mendefinisikan konstanta seperti `TRACE_DIR`, `MATCHED_UNIT_MAX_ITEM_GAP`, `LINE_OVERLAP_MIN_RATIO`.

#### 5.4.2.2 Konstanta dan Parameter Tuning
`MATCHED_UNIT_MAX_ITEM_GAP = 10` menentukan gap maksimum item_idx untuk unit dalam satu alignment, `LINE_OVERLAP_MIN_RATIO = 0.30` untuk deteksi line yang sama.

#### 5.4.2.3 Metode align_document() untuk Multi-Halaman
Mengiterasi semua halaman dan memanggil `align()` untuk setiap halaman dengan tracking cross-page `max_openxml_idx`.

#### 5.4.2.4 Metode align() sebagai Entry Point Utama
Menerima parameter: `doc_id`, `page_num`, `extraction_items`, `page_width`, `page_height`, `total_pages`, `min_openxml_idx`.

### 5.4.3 AlignmentPreprocessMixin (Preprocessing)

#### 5.4.3.1 Fungsi _flatten_extraction_items()
Mengkonversi struktur extraction hasil PDFExtractor menjadi flat list of pdf_units dengan tipe: group, table (cells), hline_table, shape, image.

#### 5.4.3.2 Transformasi Berbagai Tipe Item
Setiap tipe item diproses berbeda: grup menjadi satu unit, tabel dipecah per sel, shape yang berurutan digabung.

#### 5.4.3.3 Fungsi _extract_cell_content_text() untuk Tabel
Mengekstrak teks dari struktur cell content termasuk handling gambar dalam sel dengan placeholder [IMG:n].

#### 5.4.3.4 Fungsi _merge_consecutive_shape_items()
Shape yang berurutan digabungkan menjadi satu unit dengan merged_bbox untuk menyederhanakan alignment.

#### 5.4.3.5 Deteksi Header/Footer Zone
`_is_item_in_header_footer_zone()` mengecek apakah bbox center Y berada di zona margin atas/bawah berdasarkan section properties.

#### 5.4.3.6 Fungsi _filter_header_footer_items()
Memisahkan pdf_units yang berada di zona header/footer ke list terpisah untuk penanganan khusus.

### 5.4.4 AlignmentOpenXmlMixin (Pengambilan Data OpenXML)

#### 5.4.4.1 Fungsi _get_openxml_elements() dari Database
Query ke tabel `dokumen_elemen` dengan filter `dokumen_id` dan ordering by `delemen_sequence`.

#### 5.4.4.2 Fungsi _get_doc_sections() untuk Section Properties
Mengambil data section termasuk margin dalam twips yang digunakan untuk deteksi header/footer zone.

#### 5.4.4.3 Estimasi Page Sequence Range
`_estimate_page_sequence_range()` mengestimasi range sequence OpenXML untuk halaman tertentu berdasarkan total halaman.

#### 5.4.4.4 Deteksi Shape Content dalam JSON Tree
`_has_shape_content()` mengecek apakah element mengandung Drawing/Shape di JSON tree-nya.

#### 5.4.4.5 Ekstraksi Teks dari JSON Tree dengan _extract_text_from_json_tree()
Rekursif walk JSON tree untuk mengumpulkan teks dari semua node, mengkonversi image ke placeholder.

#### 5.4.4.6 Penanganan Image Placeholder [IMG:hash]
Gambar dikonversi ke placeholder `[IMG:abc123]` dimana hash diderivasi dari teks sekitarnya untuk matching yang lebih baik.

#### 5.4.4.7 Fungsi _build_openxml_units()
Membangun list openxml_units dari elements dengan properties: element_id, element_sequence, text, text_normalized, is_table, cells, is_shape.

### 5.4.5 AlignmentMatchingMixin (Algoritma Matching)

#### 5.4.5.1 Strategi Two-Pass Alignment
Pass pertama mencocokkan secara forward, pass kedua mencoba mencocokkan unit yang terlewat dengan OpenXML yang belum ter-match.

#### 5.4.5.2 Pass Pertama: Matching Maju
Iterasi pdf_units secara berurutan, untuk setiap unit cari OpenXML yang belum ter-match dengan similarity ratio tertinggi.

#### 5.4.5.3 Pass Kedua: Matching Elemen yang Terlewat
`_match_remaining_with_unaligned_openxml()` mencoba mencocokkan pdf_units yang tidak ter-match dengan sisa OpenXML units.

#### 5.4.5.4 Fungsi _perform_char_alignment() dengan difflib
Menggunakan `difflib.SequenceMatcher` untuk menghitung similarity ratio antara teks PDF dan teks OpenXML yang di-normalize.

#### 5.4.5.5 SequenceMatcher untuk Similarity Ratio
Threshold ratio digunakan untuk menentukan match: ratio tinggi = confident match, ratio rendah = no match.

#### 5.4.5.6 Fungsi _build_alignments_from_matching()
Mengkonversi hasil matching (openxml_to_pdf mapping) menjadi struktur alignment dengan merged_bbox, matched_pdf_units.

#### 5.4.5.7 Penanganan Table Cell Grouping
Sel-sel tabel yang ter-match ke OpenXML table element yang sama digabungkan dalam satu alignment dengan is_table=True.

#### 5.4.5.8 Tracing dan Debugging Alignment
`_append_alignment_trace()` menulis log detail ke file untuk debugging proses alignment.

### 5.4.6 AlignmentPostprocessMixin (Postprocessing)

#### 5.4.6.1 Fungsi _merge_line_overlap_alignments()
Menggabungkan alignments yang berada pada line yang sama (Y overlap ≥ 30%) menjadi satu alignment.

#### 5.4.6.2 Penggabungan Unit dengan Y Overlap
`_cluster_units_by_line()` mengelompokkan matched_pdf_units berdasarkan line untuk penanganan alignment multi-unit.

#### 5.4.6.3 Fungsi _filter_sparse_matched_units()
Menghapus unit yang terlalu jauh dari cluster utama berdasarkan item_idx gap.

#### 5.4.6.4 Filtering berdasarkan Item Gap Threshold
`MATCHED_UNIT_MAX_ITEM_GAP = 10` menentukan gap maksimum yang diizinkan antar unit dalam satu alignment.

#### 5.4.6.5 Fungsi _absorb_unaligned_by_y_overlap()
Mencoba memasukkan unit yang tidak ter-align ke alignment terdekat berdasarkan Y overlap.

#### 5.4.6.6 Fungsi _absorb_unaligned_into_alignments()
Strategi lain untuk absorpsi berdasarkan containment bbox.

#### 5.4.6.7 Resolusi Konflik Shape dengan _resolve_shape_alignment_conflicts()
Menangani kasus dimana shape unit ter-match ke multiple OpenXML elements.

#### 5.4.6.8 Cleanup Punctuation-Only Alignments
Menghapus alignment yang hanya berisi punctuation karena kemungkinan false positive.

#### 5.4.6.9 Recompute Bounding Boxes
Setelah perubahan pada matched_pdf_units, merged_bbox dihitung ulang.

### 5.4.7 Cross-Page Tracking

#### 5.4.7.1 Konsep min_openxml_idx dan max_openxml_idx
`min_openxml_idx` mencegah matching mundur ke element yang sudah di halaman sebelumnya, `max_openxml_idx` dikembalikan untuk halaman berikutnya.

#### 5.4.7.2 Pencegahan Backward Matching
Jika `new_max_openxml_idx < max_openxml_idx`, nilai lama dipertahankan dengan warning log.

#### 5.4.7.3 Kontinuitas Sequence antar Halaman
Memastikan elemen OpenXML yang sudah ter-match tidak bisa ter-match lagi di halaman berikutnya.

---

## 5.5 Klasifikasi dengan Docling

### 5.5.1 Pengenalan IBM Docling

#### 5.5.1.1 Latar Belakang dan Pengembangan Docling
Docling adalah open-source library dari IBM Research untuk document understanding, dapat mengekstrak struktur dokumen dari PDF.

#### 5.5.1.2 Kemampuan AI untuk Document Understanding
Menggunakan model deep learning untuk mengenali layout dokumen termasuk paragraf, heading, tabel, gambar, dan caption.

#### 5.5.1.3 Jenis Label yang Dihasilkan Docling
Label utama: text, section_header, table, picture, page_header, page_footer, caption, formula, footnote, list_item, code.

#### 5.5.1.4 Perbandingan dengan Model Klasifikasi Lain
Docling memiliki keunggulan dalam accuracy untuk dokumen akademis dibanding model general-purpose.

### 5.5.2 Implementasi DoclingService

#### 5.5.2.1 Struktur Class DoclingService
Class sederhana dengan `__init__` kosong dan method utama `classify_document(doc_id)`.

#### 5.5.2.2 Import DocumentConverter dari Library Docling
Menggunakan `from docling.document_converter import DocumentConverter` untuk konversi PDF.

#### 5.5.2.3 Pengambilan Path PDF dari Database
Query ke tabel Dokumen untuk mendapat `dokumen_pdf_path`, lalu gabungkan dengan `STORAGE_BASE`.

### 5.5.3 Proses Klasifikasi Dokumen

#### 5.5.3.1 Fungsi classify_document()
Entry point utama yang menerima `doc_id` dan mengembalikan dictionary dengan `predictions_by_page`.

#### 5.5.3.2 Inisialisasi DocumentConverter
`converter = DocumentConverter()` menginisialisasi converter default.

#### 5.5.3.3 Konversi PDF ke Docling Document Object
`result = converter.convert(pdf_path)` memproses seluruh PDF dan menghasilkan `docling_doc = result.document`.

### 5.5.4 Ekstraksi Koordinat dan Label

#### 5.5.4.1 Pembacaan Page Heights dengan PyMuPDF
Docling tidak menyediakan page dimensions, sehingga PyMuPDF digunakan untuk mendapatkan height setiap halaman.

#### 5.5.4.2 Fungsi process_item() untuk Setiap Elemen
Helper function yang memproses setiap item (text, table, picture) dari docling_doc.

#### 5.5.4.3 Transformasi Koordinat Docling ke PDF Points
Docling menggunakan koordinat dengan origin berbeda, perlu transformasi `y_top = page_height - bbox.t`.

#### 5.5.4.4 Inversi Koordinat Y (Bottom-Left Origin Issue)
Koordinat Y dari Docling diinversikan karena origin di bottom-left sedangkan output diharapkan top-left.

### 5.5.5 Pengolahan Berbagai Tipe Elemen

#### 5.5.5.1 Ekstraksi Texts (Paragraf, Heading)
`for text in docling_doc.texts: process_item(text)` - label diambil dari `item.label`.

#### 5.5.5.2 Ekstraksi Tables dengan Label Override
`for table in docling_doc.tables: process_item(table, label_override='table')` - memastikan tabel mendapat label yang benar.

#### 5.5.5.3 Ekstraksi Pictures dan Formulas
Pictures dan formulas diekstrak dari attributes khusus `docling_doc.pictures` dan `docling_doc.formulas` jika tersedia.

### 5.5.6 Struktur Output Klasifikasi

#### 5.5.6.1 Pengelompokan predictions_by_page
Output adalah dictionary dengan key berupa string page number ("1", "2", ...) dan value berupa list predictions.

#### 5.5.6.2 Sorting Prediksi berdasarkan Posisi
Setiap page predictions diurutkan: `sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))` - Y first, then X.

#### 5.5.6.3 Format Return Value
`{'success': True, 'total_pages': n, 'bbox_unit': 'pdf_points', 'predictions_by_page': {...}}`.

### 5.5.7 Label-label Docling

#### 5.5.7.1 text - Paragraf Teks Biasa
Label default untuk konten teks yang tidak memiliki karakteristik khusus.

#### 5.5.7.2 section_header - Judul Section
Untuk heading dan judul section, biasanya dengan font lebih besar atau bold.

#### 5.5.7.3 table - Area Tabel
Menandai rectangular area yang berisi tabel dengan cells.

#### 5.5.7.4 picture - Area Gambar
Untuk gambar, diagram, chart, dan visual non-teks lainnya.

#### 5.5.7.5 page_header dan page_footer
Elemen yang berada di zona header (atas) atau footer (bawah) halaman.

#### 5.5.7.6 caption - Keterangan Gambar/Tabel
Teks deskriptif yang biasanya berada di bawah gambar atau tabel dengan awalan "Gambar" atau "Tabel".

#### 5.5.7.7 formula - Rumus Matematika
Ekspresi matematika yang biasanya centered dan standalone.

#### 5.5.7.8 footnote - Catatan Kaki
Teks di bagian bawah halaman dengan nomor referensi.

#### 5.5.7.9 list_item - Item dalam List
Elemen dalam unnumbered atau numbered list.

---

## 5.6 Proses Fusion (Penggabungan Alignment dan Docling)

### 5.6.1 Konsep Fusion dalam Pipeline

#### 5.6.1.1 Tujuan Menggabungkan Alignment dengan Klasifikasi
Alignment menyediakan mapping ke OpenXML element, Docling menyediakan label visual. Fusion menggabungkan keduanya.

#### 5.6.1.2 Prioritas Sumber Data (Alignment vs Docling)
Alignment adalah sumber utama untuk element_id, Docling memberikan label. Jika keduanya ada, Docling label digunakan dengan koreksi.

#### 5.6.1.3 Penanganan Konflik dan Overlap
Ketika bbox alignment overlap dengan multiple Docling predictions, yang dengan overlap ratio tertinggi dipilih.

### 5.6.2 Implementasi DoclingFusionService

#### 5.6.2.1 Struktur Class DoclingFusionService
Class dengan `__init__` menerima optional `section_data` untuk margin configuration.

#### 5.6.2.2 Inisialisasi dengan Section Data
Section data berisi `page_height_pt`, `margin_top_pt`, `margin_bottom_pt` untuk deteksi header/footer zone.

#### 5.6.2.3 Parameter Page Height dan Margins
Default: `page_height_pt=842` (A4), `margin_top_pt=72`, `margin_bottom_pt=72` (1 inch).

### 5.6.3 Perhitungan Overlap Bounding Box

#### 5.6.3.1 Fungsi calculate_overlap()
Menghitung intersection area dari dua bbox kemudian membagi dengan area bbox yang lebih kecil.

#### 5.6.3.2 Metrik Intersection-over-Minimum-Area
Formula: `overlap_ratio = intersection_area / min(area1, area2)` - berbeda dari IoU untuk menangani contained boxes.

#### 5.6.3.3 Threshold untuk Penentuan Match
Overlap ratio > 0 berarti ada overlap; semakin tinggi semakin confident match.

### 5.6.4 Deteksi Margin Zone

#### 5.6.4.1 Fungsi get_bbox_margin_zone()
Menghitung center Y bbox dan mengecek apakah berada di zona header (< margin_top) atau footer (> page_height - margin_bottom).

#### 5.6.4.2 Penentuan Area Header dan Footer
Header zone: Y < margin_top_pt. Footer zone: Y > page_height_pt - margin_bottom_pt.

#### 5.6.4.3 Penggunaan Section Properties untuk Margin
Jika tersedia, margin dari DokumenSection digunakan; jika tidak, default 72pt (1 inch) digunakan.

### 5.6.5 Koreksi Label Header dan Footer

#### 5.6.5.1 Fungsi correct_header_footer_label()
Mengecek apakah label page_header/page_footer sesuai dengan posisi bbox di margin zone.

#### 5.6.5.2 Koreksi Label yang Salah Posisi
Jika Docling memberikan label page_header tapi bbox tidak di zona header, label dikoreksi ke 'text'.

#### 5.6.5.3 Fallback ke Label 'text'
Default fallback untuk elemen yang labelnya tidak sesuai dengan posisinya.

### 5.6.6 Fallback Label dari Alignment

#### 5.6.6.1 Fungsi fallback_label()
Menentukan label jika tidak ada match Docling, berdasarkan element_type dari alignment.

#### 5.6.6.2 Penentuan Label berdasarkan element_type
Mapping: Paragraph → 'text', Table → 'table', Drawing → 'picture', dll.

#### 5.6.6.3 Penanganan Zona Header/Footer
Jika alignment memiliki zone='header'/'footer', label page_header/page_footer digunakan.

### 5.6.7 Deteksi Area Gambar

#### 5.6.7.1 Fungsi is_picture_area()
Mengecek apakah item berisi gambar atau shape berdasarkan properties seperti `has_image`, `is_shape`.

#### 5.6.7.2 Pengecekan Image dan Shape Content
Item dengan `is_image=True` atau `is_shape=True` dianggap picture area.

### 5.6.8 Proses Fusion Utama

#### 5.6.8.1 Fungsi fuse_alignments_with_docling()
Method utama yang menerima `alignments`, `header_footer_units`, `docling_predictions` dan mengembalikan list fused results.

#### 5.6.8.2 Iterasi pada Setiap Alignment
Setiap alignment diproses: mencari best match Docling, menentukan label, membangun fused item.

#### 5.6.8.3 Pencarian Best Match dengan Docling
Untuk setiap alignment bbox, cari Docling prediction dengan overlap ratio tertinggi.

#### 5.6.8.4 Penggabungan Bounding Box dengan merge_bboxes()
Static method yang menggabungkan dua bbox menjadi encompassing bbox: `[min(x0), min(y0), max(x1), max(y1)]`.

### 5.6.9 Penanganan Caption

#### 5.6.9.1 Deteksi Caption Candidate dengan _is_caption_candidate()
Mengecek teks untuk pola caption seperti "Gambar 1.", "Tabel 2.", "Sumber:".

#### 5.6.9.2 Pengecekan Item Above/Below
`_has_item_above()` dan `_has_item_below()` mengecek apakah ada picture/table di atas/bawah caption.

#### 5.6.9.3 Pengaitan Caption dengan Picture/Table
Caption diasosiasikan dengan picture/table terdekat untuk pemberian label struktural.

### 5.6.10 Finalisasi dan Pengurutan

#### 5.6.10.1 Sorting Hasil Fusion
`sort_key(item)` dan `compare(a, b)` digunakan untuk mengurutkan berdasarkan Y kemudian X.

#### 5.6.10.2 Penanganan Header/Footer Units
Header/footer units ditambahkan ke hasil dengan label yang sesuai (page_header/page_footer).

#### 5.6.10.3 Struktur Output Fused Results
List of dict dengan keys: `element_id`, `bbox`, `label`, `docling_label`, `text`, `source`, dll.

---

## 5.7 MergingExtractionService (Pemberian Label dan Penyesuaian)

### 5.7.1 Peran MergingExtractionService sebagai Orkestrator

#### 5.7.1.1 Koordinasi Seluruh Pipeline
Service ini adalah entry point yang memanggil semua service lain secara berurutan: Docling → PDFExtractor → Alignment → Fusion.

#### 5.7.1.2 Penggabungan Semua Service
`__init__` menginisialisasi: `alignment_service`, `docling_service`, `fusion_service`, `visualization_service`.

#### 5.7.1.3 Manajemen Alur Data End-to-End
Mengelola flow data dari ekstraksi sampai penyimpanan ke database.

### 5.7.2 Inisialisasi dan Dependencies

#### 5.7.2.1 Import Services
Mengimport: `PDFExtractor`, `AlignmentService`, `DoclingService`, `DoclingFusionService`, `VisualizationService`.

#### 5.7.2.2 Inisialisasi Service Instances
Di `__init__`: membuat instance dari setiap service untuk digunakan di seluruh class.

#### 5.7.2.3 Konfigurasi Environment Variables
`STORAGE_BASE` dan `VISUALIZATION_OUTPUT` diambil dari environment dengan default values.

### 5.7.3 Fungsi process_document() Utama

#### 5.7.3.1 Validasi Dokumen dari Database
Query `Dokumen` by `doc_id`, return False jika tidak ditemukan.

#### 5.7.3.2 Pengambilan PDF Path
Path PDF diambil dari `dokumen.dokumen_pdf_path` dan digabung dengan `STORAGE_BASE`.

#### 5.7.3.3 Eksekusi Docling pada Level Dokumen
`docling_service.classify_document(doc_id)` dijalankan sekali untuk seluruh dokumen, hasilnya di-cache.

#### 5.7.3.4 Iterasi per Halaman
Loop `for page_num in range(1, total_pages + 1):` memproses setiap halaman.

### 5.7.4 Transformasi Data Ekstraksi

#### 5.7.4.1 Fungsi _transform_extraction_data_to_items()
Mengkonversi dictionary hasil PDFExtractor ke list items dengan format yang diharapkan AlignmentService.

#### 5.7.4.2 Konversi char_groups ke Type 'group'
Setiap char_group menjadi `{'type': 'group', 'bbox': ..., 'data': {'text': ...}}`.

#### 5.7.4.3 Konversi basic_tables dan hline_tables
Tabel menjadi `{'type': 'table'/'hline_table', 'bbox': ..., 'data': {'rows': ...}}`.

#### 5.7.4.4 Konversi shapes dan images
Shapes dan images menjadi `{'type': 'shape'/'image', 'bbox': ..., 'data': {...}}`.

#### 5.7.4.5 Sorting dengan Reading Order Awareness
`compare_items()` dengan Y overlap 30% threshold untuk menentukan same-line.

### 5.7.5 Penanganan Footnote

#### 5.7.5.1 Konstanta FOOTNOTE_LABELS dan Threshold
`FOOTNOTE_LABELS = {"footnote"}`, `FOOTNOTE_MATCH_MIN_RATIO = 0.55`, `FOOTNOTE_OVERLAP_THRESHOLD = 0.3`.

#### 5.7.5.2 Fungsi _build_footnote_groups()
Mendeteksi area footnote berdasarkan Docling predictions dengan label 'footnote'.

#### 5.7.5.3 Matching Footnote dengan Referensi dalam Teks
Menggunakan difflib untuk mencocokkan teks footnote dengan referensi superscript di body text.

#### 5.7.5.4 Fungsi _assign_docling_footnotes()
Mengasosiasikan footnote predictions dengan DokumenNote di database.

### 5.7.6 Deteksi Duplikasi OpenXML Element

#### 5.7.6.1 Fungsi _collect_duplicate_openxml_element_ids()
Mengumpulkan element_id yang muncul di lebih dari satu halaman.

#### 5.7.6.2 Identifikasi Element yang Muncul di Multiple Pages
Iterasi semua alignments, track element_id per page, return set of duplicates.

#### 5.7.6.3 Konstanta DUPLICATE_SEQUENCE_GAP_THRESHOLD
Nilai 2 menentukan gap sequence minimal untuk dianggap terpisah (bukan duplikat yang berdekatan).

### 5.7.7 Merge Duplicate Units dengan Neighbors

#### 5.7.7.1 Fungsi _merge_duplicate_units_with_neighbors()
Menggabungkan unit dari duplicate elements ke alignment tetangga (above/below) yang memiliki teks yang sama.

#### 5.7.7.2 Pengecekan Text Containment
Mengecek apakah teks unit terkandung dalam teks alignment above/below.

#### 5.7.7.3 Pemilihan Target (Above vs Below)
Jika keduanya mengandung teks, pilih yang lebih dekat berdasarkan Y distance.

#### 5.7.7.4 Recompute Bounding Boxes
Setelah merge, bounding box alignment yang ter-affect di-recompute.

### 5.7.8 Sinkronisasi Fused Bboxes dengan Alignments

#### 5.7.8.1 Fungsi _sync_fused_bboxes_with_alignments()
Menyinkronkan bbox di fused_results dengan bbox dari alignments yang sudah di-update.

#### 5.7.8.2 Penghapusan Elemen yang Removed
Fused results dengan element_id yang ada di removed_element_ids dihapus.

#### 5.7.8.3 Update Bounding Box
Fused items yang source='alignment' diupdate bbox-nya dari alignment terbaru.

### 5.7.9 Pemberian Label Struktural

#### 5.7.9.1 Fungsi _apply_structural_labels()
Mengiterasi fused_results dan menambahkan field `dev_label_struktural` berdasarkan analisis lebih lanjut.

#### 5.7.9.2 Deteksi Judul BAB dengan Center Alignment
Jika `section_header` + center-aligned + teks dimulai "BAB" → label 'judul_bab'.

#### 5.7.9.3 Deteksi Judul Subbab dengan Pola Numerik
Regex `^\s*\d+(?:\.\d+)+\.?` untuk mendeteksi pola seperti "1.1", "2.3.1" → label 'judul_subbab'.

#### 5.7.9.4 Penanganan Caption (caption_tabel, caption_gambar)
Caption di-relabel ke 'caption_tabel' atau 'caption_gambar' berdasarkan kedekatan dengan tabel atau gambar.

#### 5.7.9.5 Deteksi List dengan Level Tracking
Regex untuk numeric (1.), alpha (a.), dan bullet markers. Level di-track dengan `list_marker_levels` dict.

### 5.7.10 Pengecekan Alignment Paragraf

#### 5.7.10.1 Fungsi _is_paragraph_center_aligned()
Memeriksa apakah paragraf menggunakan center alignment dari OpenXML properties.

#### 5.7.10.2 Ekstraksi dari JSON Tree
`_extract_paragraph_alignment()` mencari key 'jc', 'alignment', dll di JSON tree.

#### 5.7.10.3 Pengambilan dari dokumen_format_paragraf
Jika tidak ada di JSON tree, query ke tabel dokumen_format_paragraf untuk dfp_jc.

### 5.7.11 Pengecekan Bold State

#### 5.7.11.1 Fungsi _get_element_bold_state()
Memeriksa apakah text runs dalam element menggunakan bold formatting.

#### 5.7.11.2 Ekstraksi text_run_ids
`_extract_text_run_ids()` mengumpulkan dftx_id dari JSON tree.

#### 5.7.11.3 Query ke dokumen_format_text
Batch query untuk dftx_bold dari tabel dokumen_format_text.

### 5.7.12 Expansion Caption Label

#### 5.7.12.1 Propagasi Label ke Baris Berikutnya
Jika caption multi-baris, label caption dipropagasi ke baris berikutnya yang formatnya sama.

#### 5.7.12.2 Pencocokan Formatting (Alignment, Bold)
Baris berikutnya harus memiliki alignment dan bold state yang sama dengan caption pertama.

#### 5.7.12.3 Kondisi Berhenti Expansion
Berhenti jika: label bukan text/section_header, formatting berbeda, atau tidak ada element berikutnya.

### 5.7.13 Visualisasi dan Output JSON

#### 5.7.13.1 Pemanggilan VisualizationService
Jika `generate_visualizations=True`, `visualization_service.visualize_page()` dipanggil untuk setiap halaman.

#### 5.7.13.2 Parameter visualize_page()
Menerima: pdf_path, page_num (0-based), alignments, fused_results, header_footer_units, unaligned_pdf_units, duplicate_mapping_units.

#### 5.7.13.3 Penyimpanan JSON fusion_data per Halaman
File `page_{n}_fusion_data.json` disimpan dengan alignments, fused_results, raw_docling untuk debugging.

---

## 5.8 Penyimpanan di Database

### 5.8.1 Model Data untuk Analisa Visual

#### 5.8.1.1 Model DokumenElemenVisual
Tabel `dokumen_elemen_visual` menyimpan hasil analisa visual dengan kolom: dev_id (PK), dokumen_id, dev_page, dokumen_elemen_id (FK), dev_bbox_x0/y0/x1/y1, dev_label, dev_label_struktural, dev_text.

#### 5.8.1.2 Relasi dengan Model DokumenElemen
`dokumen_elemen_id` adalah foreign key ke tabel `dokumen_elemen` yang berisi data OpenXML.

#### 5.8.1.3 Field-field Utama
`dev_label`: label dari Docling (text, table, picture, etc). `dev_label_struktural`: label derived (judul_bab, caption_gambar, etc).

### 5.8.2 Fungsi _save_alignment_results()

#### 5.8.2.1 Parameter Input
Menerima: db session, alignments, docling_predictions, footnote_entries, header_footer_units, section_data, doc_id, page_num.

#### 5.8.2.2 Pemanggilan Fusion Service
`self.fusion_service.fuse_alignments_with_docling()` dipanggil untuk menggabungkan data.

#### 5.8.2.3 Persiapan Data untuk Database
Fused results dikonversi ke format yang siap disimpan ke DokumenElemenVisual.

### 5.8.3 Fungsi _replace_visual_records()

#### 5.8.3.1 Penghapusan Records Lama per Halaman
`DELETE FROM dokumen_elemen_visual WHERE dokumen_id=? AND dev_page=?` sebelum insert baru.

#### 5.8.3.2 Bulk Insert Records Baru
Iterasi fused_results dan create DokumenElemenVisual untuk setiap item.

#### 5.8.3.3 Penggunaan Transaction dan Commit
Commit dilakukan di akhir `process_document()` setelah semua halaman selesai.

### 5.8.4 Struktur Field DokumenElemenVisual

#### 5.8.4.1 dev_id (Primary Key)
Auto-increment BigInteger primary key.

#### 5.8.4.2 dokumen_id dan dev_page
Integer fields untuk identifikasi dokumen dan nomor halaman (1-based).

#### 5.8.4.3 dokumen_elemen_id (Foreign Key ke OpenXML Element)
BigInteger referencing `dokumen_elemen.delemen_id`, nullable untuk elemen yang tidak ter-align.

#### 5.8.4.4 dev_bbox_x0/y0/x1/y1 (Koordinat Bounding Box)
Float fields untuk koordinat dalam PDF points.

#### 5.8.4.5 dev_label
String(50) untuk label dari Docling: text, section_header, table, picture, caption, etc.

#### 5.8.4.6 dev_label_struktural
String(50) untuk label struktural derived: judul_bab, judul_subbab, paragraf, caption_gambar, list_level_1, etc.

#### 5.8.4.7 dev_text
Text field untuk konten teks dari elemen.

### 5.8.5 Model Pendukung Lainnya

#### 5.8.5.1 Model Dokumen
Menyimpan metadata dokumen: dokumen_id, mhs_nrp, dokumen_pdf_path, dokumen_images_path, dll.

#### 5.8.5.2 Model DokumenElemen
Menyimpan elemen OpenXML: delemen_id, dokumen_id, delemen_type, delemen_text, delemen_json_tree, delemen_sequence.

#### 5.8.5.3 Model DokumenSection
Menyimpan section properties: dsec_page_width_twips, dsec_page_height_twips, margins dalam twips.

#### 5.8.5.4 Model DokumenNote untuk Footnote
Menyimpan footnote yang sudah di-extract dari OpenXML.

### 5.8.6 Model Antrian

#### 5.8.6.1 Struktur Tabel Antrian
Kolom: antrian_id (PK), antrian_tipe (Enum: dokumen/buku), buku_id, bab_id, dokumen_id, status fields, timestamps.

#### 5.8.6.2 Field antrian_labeling_status
Enum: 'in_queue', 'processing', 'completed', 'failed' untuk tracking status labeling visual.

#### 5.8.6.3 Field antrian_validation_status
Enum sama untuk tracking status validasi setelah labeling selesai.

#### 5.8.6.4 Transisi Status dalam Pipeline
Alur: extraction → labeling (in_queue → processing → completed) → validation (in_queue → processing → completed).

### 5.8.7 Konfigurasi Database

#### 5.8.7.1 Penggunaan SQLAlchemy ORM
Semua model mewarisi dari `Base` yang didefinisikan di `database.py`.

#### 5.8.7.2 SessionLocal dan Engine Configuration
`SessionLocal = sessionmaker(bind=engine)` untuk membuat database sessions.

#### 5.8.7.3 Environment Variables untuk Connection String
`DATABASE_URL` dari environment untuk PostgreSQL connection string.

### 5.8.8 Integrasi dengan Service Validasi

#### 5.8.8.1 Data yang Dibutuhkan oleh Service Validasi
Service validasi mengakses `dokumen_elemen_visual` untuk mendapatkan posisi dan label setiap elemen.

#### 5.8.8.2 Query Pattern untuk Pengambilan DokumenElemenVisual
`SELECT * FROM dokumen_elemen_visual WHERE dokumen_id=? ORDER BY dev_page, dev_bbox_y0`.

#### 5.8.8.3 Format yang Compatible dengan Validation Rules
Label struktural seperti judul_bab, judul_subbab digunakan untuk validasi hierarki dan formatting.

---

*Dokumen ini merupakan daftar isi lengkap untuk BAB 5 dengan penjelasan singkat setiap item berdasarkan kode di repository ai-service-python.*
