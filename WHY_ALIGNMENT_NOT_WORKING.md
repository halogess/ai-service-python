# KENAPA ALIGNMENT TIDAK BERJALAN

## ROOT CAUSE

**Alignment membutuhkan DokumenElemen dari database, tapi DokumenElemen KOSONG atau TIDAK ADA.**

## ALUR LENGKAP

### 1. **DokumenElemen Dibuat Dari DOCX**
```
DOCX File → OpenXML Parser → DokumenElemen (database)
```

DokumenElemen berisi:
- `delemen_id`: ID element
- `delemen_sequence`: Urutan element
- `delemen_type`: Tipe (paragraph, table, dll)
- `delemen_json_tree`: Struktur JSON dari OpenXML
- `delemen_text`: Text content

### 2. **Alignment Mencocokkan PDF dengan DokumenElemen**
```
PDF Text (extraction) ↔ DokumenElemen Text (database)
         ↓
    Difflib Matching
         ↓
   Alignment Result
```

### 3. **Jika DokumenElemen Kosong**
```
PDF Text → ❌ No Match → Alignment Kosong → Visualisasi Kosong
```

## KENAPA DOKUMEN ELEMEN KOSONG?

### Kemungkinan 1: **Database Tidak Terisi**
- Belum ada proses upload DOCX
- Belum ada parsing OpenXML
- Database baru/kosong

### Kemungkinan 2: **Database Tidak Tersambung**
```
Can't connect to MySQL server on 'host.docker.internal'
```

### Kemungkinan 3: **Dokumen ID Salah**
- `task.dokumen_id` NULL atau tidak valid
- Dokumen ada tapi tidak punya DokumenElemen

## SOLUSI

### ✅ Opsi 1: Start Database & Upload DOCX
```bash
# 1. Start database
docker-compose up -d

# 2. Upload DOCX file melalui Flask app
# 3. Parse DOCX → DokumenElemen
# 4. Run alignment
```

### ✅ Opsi 2: Cek Database Connection
```bash
# Check .env
type .env | findstr DB

# Test connection
python -c "from database import SessionLocal; db = SessionLocal(); print('OK')"
```

### ✅ Opsi 3: Cek DokumenElemen
```sql
-- Cek jumlah dokumen
SELECT COUNT(*) FROM dokumen;

-- Cek jumlah elemen untuk dokumen tertentu
SELECT COUNT(*) 
FROM dokumen_elemen de
JOIN dokumen_part dp ON de.dpart_id = dp.dpart_id
JOIN dokumen_section ds ON dp.dsec_id = ds.dsec_id
WHERE ds.dokumen_id = 1 AND dp.dpart_type = 'body';
```

### ✅ Opsi 4: Simulasi Tanpa Database
```bash
# Gunakan mock alignment (tidak butuh database)
python src/tests/simulate_bab2.py
```

## KESIMPULAN

**Alignment tidak berjalan karena:**

1. ❌ **Database tidak connect** → Tidak bisa ambil DokumenElemen
2. ❌ **DokumenElemen kosong** → Tidak ada data untuk di-match
3. ❌ **Belum upload DOCX** → DokumenElemen belum dibuat

**BUKAN karena:**
- ✅ Visualizer (sudah terbukti bekerja)
- ✅ PDF Extraction (sudah berhasil extract 58 items)
- ✅ Alignment logic (sudah benar)

**Yang perlu dilakukan:**
1. **Start database** (docker-compose up -d)
2. **Upload DOCX file** untuk membuat DokumenElemen
3. **Run alignment** dengan dokumen_id yang valid

---

**Catatan:** Alignment adalah proses **matching PDF text dengan DOCX text**. Tanpa DOCX (DokumenElemen), alignment tidak bisa berjalan!
