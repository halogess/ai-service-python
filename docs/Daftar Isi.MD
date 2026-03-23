# BAB 5 ANALISIS VISUAL DOKUMEN DAN EVALUASI AKURASI

Bab ini membahas layanan analisis visual dokumen yang dijalankan setelah ekstraksi struktur dokumen dan sebelum validasi. Fokus utama bab ini adalah proses pembentukan elemen visual dari PDF, penggunaan Docling sebagai komponen AI dokumen, serta penyelarasan hasil visual dengan struktur Open XML. Selain itu, bab ini juga membahas evaluasi akurasi untuk menilai kualitas pelabelan visual dan dampaknya terhadap proses validasi.

## 5.1 Layanan Analisis Visual Dokumen

Layanan analisis visual dokumen merupakan komponen yang bertugas membangun representasi visual elemen dokumen berdasarkan tampilan PDF. Layanan ini dijalankan oleh worker terpisah sehingga pemrosesan tidak bergantung pada sesi pengguna di frontend. Hasil akhirnya berupa elemen visual yang memiliki label, teks ringkas, dan koordinat bounding box yang siap dipakai pada tahap validasi.

### 5.1.1 Peran Layanan dalam Pipeline Sistem

Subbab ini menjelaskan kedudukan layanan analisis visual dalam keseluruhan alur sistem. Layanan ini menjadi penghubung antara hasil ekstraksi struktur dokumen dan proses validasi berbasis aturan. Tanpa tahap ini, sistem hanya memiliki struktur logis dokumen tetapi belum memiliki lokasi visual yang presisi pada halaman PDF.

#### 5.1.1.1 Posisi setelah ekstraksi struktur dokumen

Layanan analisis visual dijalankan setelah main service selesai mengekstraksi struktur DOCX ke database dan mengonversi dokumen ke PDF. Pada tahap ini, setiap dokumen atau bab sudah memiliki data elemen Open XML dan path PDF yang dapat dipakai sebagai masukan. Dengan demikian, worker visual tidak memulai dari file mentah, tetapi dari representasi dokumen yang sudah siap dipetakan secara visual.

#### 5.1.1.2 Keterkaitan dengan main service dan basis data

Main service bertugas membuat antrian, menyimpan hasil ekstraksi struktur, dan menyediakan metadata dokumen yang diperlukan worker visual. Worker visual pada `ai-service-python` membaca data tersebut langsung dari basis data yang sama dan mengambil file dari storage bersama. Pola ini membuat integrasi antarlayanan tidak memerlukan pemanggilan API langsung selama pemrosesan background berlangsung.

#### 5.1.1.3 Kontribusi terhadap validasi dan frontend

Keluaran layanan ini dipakai oleh worker validasi untuk mengetahui lokasi visual elemen yang sedang diperiksa. Data yang sama juga dipakai frontend untuk menampilkan posisi kesalahan pada halaman dokumen secara lebih mudah dipahami pengguna. Karena itu, layanan visual berfungsi sekaligus sebagai dasar validasi dan dasar visualisasi hasil.

### 5.1.2 Mekanisme Pemrosesan Asinkron

Subbab ini membahas bagaimana layanan visual bekerja melalui antrian pada database. Pendekatan asinkron dipilih agar unggahan dokumen tidak harus menunggu seluruh analisis visual selesai di sisi pengguna. Selain itu, mekanisme ini memudahkan pemisahan tahap ekstraksi, pelabelan visual, dan validasi ke dalam worker yang berbeda.

#### 5.1.2.1 Penggunaan antrian labeling

Worker visual secara berkala memeriksa tabel antrian untuk mencari entri dengan `antrian_labeling_status = in_queue`. Seleksi ini memastikan hanya dokumen yang sudah selesai pada tahap ekstraksi PDF yang dapat masuk ke tahap analisis visual. Dengan pendekatan tersebut, worker visual memproses pekerjaan secara berurutan sesuai data antrian yang tersedia.

#### 5.1.2.2 Perubahan status pekerjaan

Saat pekerjaan diambil, status labeling diubah menjadi `processing` untuk menandai bahwa dokumen sedang dianalisis oleh worker visual. Jika seluruh proses berhasil, status tersebut diubah menjadi `completed`, sedangkan kegagalan akan dicatat sebagai `failed` beserta pesan galatnya. Skema status ini penting untuk menjaga keterlacakan pipeline dan mencegah dokumen diproses ganda.

#### 5.1.2.3 Sinkronisasi ke tahap validasi

Setelah labeling selesai, worker visual tidak langsung melakukan validasi, tetapi meneruskan dokumen ke tahap berikutnya melalui status antrian. Implementasi saat ini mengubah `antrian_validation_status` menjadi `in_queue` agar validation worker pada main service dapat mengambilnya. Dengan desain ini, setiap tahap tetap terpisah namun tetap tersambung secara konsisten melalui basis data.

### 5.1.3 Masukan dan Keluaran Layanan

Subbab ini menjelaskan data apa saja yang digunakan worker visual dan keluaran apa yang dihasilkan. Masukan berasal dari dua sumber utama, yaitu PDF hasil konversi dan elemen struktur hasil ekstraksi Open XML. Keluaran akhirnya adalah representasi elemen visual yang lebih kaya daripada keluaran mentah Docling maupun PDF extractor.

#### 5.1.3.1 PDF hasil konversi sebagai masukan visual

PDF hasil konversi adalah representasi dokumen yang dipakai untuk menangkap tampilan aktual di halaman. Pada file inilah posisi teks, gambar, tabel, dan elemen layout lain dianalisis secara spasial. Pemakaian PDF penting karena validasi visual harus mengacu pada hasil akhir yang benar-benar dilihat pengguna, bukan hanya struktur internal DOCX.

#### 5.1.3.2 Elemen hasil ekstraksi struktur sebagai acuan

Data elemen Open XML yang tersimpan di database menjadi acuan urutan, jenis, dan isi logis dokumen. Informasi ini diperlukan karena hasil visual murni dari PDF belum tentu dapat langsung menjelaskan fungsi struktural setiap elemen. Dengan memanfaatkan kedua sumber data tersebut, sistem dapat menghubungkan struktur logis dan tampilan fisik dokumen.

#### 5.1.3.3 Elemen visual final sebagai keluaran

Keluaran worker visual disimpan sebagai elemen visual yang memuat `label`, `bbox`, halaman, dan hubungan ke elemen dokumen. Hasil ini tidak lagi berupa prediksi mentah, tetapi sudah melewati alignment dan fusion dengan struktur dokumen. Karena itu, keluaran akhir lebih siap dipakai untuk validasi aturan, pelacakan kesalahan, dan penampilan pada frontend.

## 5.2 Analisis Dokumen AI dengan Docling

Bagian ini membahas peran Docling sebagai komponen AI dokumen pada implementasi akhir sistem. Docling dipakai untuk mengenali jenis blok layout langsung dari dokumen PDF dan menghasilkan prediksi visual per halaman. Hasil dari Docling kemudian dipadukan dengan hasil ekstraksi PDF dan data struktur dokumen agar akurasinya lebih sesuai kebutuhan sistem.

### 5.2.1 Dasar Pemilihan Model

Subbab ini menjelaskan mengapa pendekatan awal berbasis LayoutLMv3 diganti dalam implementasi akhir. Pergantian ini bukan semata karena perubahan teknologi, tetapi karena kebutuhan sistem terhadap hasil visual yang lebih stabil dan lebih mudah diintegrasikan. Dengan demikian, pemilihan model dibahas sebagai keputusan desain yang berkaitan langsung dengan akurasi dan keandalan pipeline.

#### 5.2.1.1 Kedudukan LayoutLMv3 pada rancangan awal

Pada tahap rancangan awal, LayoutLMv3 dipertimbangkan karena model tersebut dikenal kuat untuk pemrosesan dokumen multimodal. Model ini secara teoritis mampu menggabungkan informasi teks, tata letak, dan citra dalam satu representasi. Oleh sebab itu, LayoutLMv3 sempat ditempatkan sebagai kandidat utama untuk tugas klasifikasi elemen dokumen.

#### 5.2.1.2 Keterbatasan akurasi pendekatan sebelumnya

Dalam konteks sistem ini, pendekatan awal belum memberikan akurasi yang cukup stabil untuk berbagai variasi dokumen tugas akhir. Permasalahan muncul terutama pada konsistensi label antarhalaman, ketepatan untuk elemen khusus, dan kebutuhan penyesuaian data yang cukup besar. Kondisi tersebut membuat pendekatan awal kurang efisien untuk dijadikan komponen operasional utama pada pipeline visual.

#### 5.2.1.3 Alasan penggunaan Docling pada implementasi akhir

Docling dipilih karena lebih sesuai dengan kebutuhan analisis struktur dokumen berbasis PDF pada sistem ini. Integrasinya lebih sederhana karena model dapat dipanggil langsung terhadap PDF tanpa menyiapkan pipeline pelatihan khusus seperti pada model terdahulu. Selain itu, hasil awal Docling cukup baik untuk dijadikan kerangka label visual yang kemudian diperkuat lagi melalui alignment dan fusion.

### 5.2.2 Proses Inferensi Docling

Subbab ini membahas bagaimana Docling digunakan pada implementasi sistem saat ini. Worker visual menjalankan Docling terhadap file PDF dan mengumpulkan hasil prediksi per halaman. Dengan cara ini, analisis visual dapat memanfaatkan label layout langsung dari keluaran model tanpa harus menunggu tahap lain terlebih dahulu.

#### 5.2.2.1 Pembacaan dokumen PDF

Implementasi `DoclingService` memanggil `DocumentConverter()` dan menjalankan `convert(pdf_path)` terhadap file PDF dokumen. Proses ini menghasilkan objek dokumen yang berisi kumpulan elemen hasil analisis seperti teks, tabel, gambar, dan formula. Dengan demikian, sumber analisis Docling adalah PDF itu sendiri, bukan citra yang dirender terpisah.

#### 5.2.2.2 Analisis elemen per halaman

Setelah dokumen terbaca, sistem mengiterasi setiap hasil Docling dan memetakan elemen ke halaman asalnya berdasarkan informasi provenance. Pada tahap ini, sistem juga mengonversi koordinat Docling ke sistem koordinat halaman PDF yang konsisten dengan tahapan berikutnya. Hasil akhirnya adalah daftar prediksi yang telah dikelompokkan per halaman sehingga lebih mudah dipadukan dengan extractor dan alignment.

#### 5.2.2.3 Pembentukan prediksi label visual

Setiap prediksi disimpan dalam bentuk struktur yang memuat teks, label, bounding box, dan sumber prediksi. Label yang dibentuk mencakup elemen seperti `text`, `table`, `picture`, dan `formula`, serta dapat diperluas pada tahap berikutnya. Bentuk keluaran ini penting karena menjadi bahan baku fusion, bukan hasil final yang langsung dipakai sistem.

### 5.2.3 Keterbatasan Hasil Awal Docling

Subbab ini menekankan bahwa keluaran Docling belum cukup untuk langsung dijadikan hasil visual final. Walaupun model mampu mengenali blok layout, hasilnya masih perlu diselaraskan dengan struktur dokumen dan posisi PDF yang lebih detail. Oleh karena itu, pembahasan ini menunjukkan alasan teknis mengapa tahap alignment dan fusion tetap diperlukan.

#### 5.2.3.1 Overlap antar bounding box

Bounding box dari Docling dapat saling tumpang tindih karena model bekerja pada tingkat blok layout yang tidak selalu eksklusif. Kasus ini terutama dapat muncul pada area padat seperti caption yang berdekatan dengan gambar atau footnote di bagian bawah halaman. Jika hasil ini dipakai secara langsung, sistem berisiko memberi label ganda atau salah pada elemen yang sama.

#### 5.2.3.2 Ketidaklengkapan area deteksi

Pada beberapa dokumen, ada elemen yang tidak terdeteksi lengkap atau hanya tertangkap sebagian oleh Docling. Hal ini dapat disebabkan oleh struktur visual yang kecil, pola layout yang tidak umum, atau keterbatasan model dalam menangkap elemen campuran. Akibatnya, hasil mentah Docling belum cukup andal untuk dijadikan dasar koordinat final tanpa bantuan tahap lain.

#### 5.2.3.3 Kebutuhan penyempurnaan pada tahap berikutnya

Karena keluaran awal Docling masih bersifat murni visual, hasil tersebut belum memiliki hubungan eksplisit dengan elemen Open XML di database. Sistem kemudian memanfaatkan alignment untuk menemukan pasangan elemen struktur dan memanfaatkan fusion untuk menetapkan label final. Dengan demikian, Docling berperan sebagai sumber label visual awal yang diperkuat oleh konteks struktur dokumen.

## 5.3 Ekstraksi Unit Visual dari PDF

Bagian ini membahas proses ekstraksi detail dari PDF yang menjadi sumber koordinat presisi pada sistem. Jika Docling berperan memberi kerangka label visual, maka PDF extractor berperan memberi unit visual yang lebih rinci dan lebih dekat dengan tampilan aktual halaman. Kombinasi keduanya membuat sistem dapat memperoleh label yang bermakna sekaligus posisi yang lebih akurat.

### 5.3.1 Pembacaan Struktur Halaman PDF

Subbab ini menjelaskan bagaimana sistem membaca PDF pada tingkat halaman. Pembacaan ini diperlukan agar seluruh proses visual menggunakan acuan ukuran dan koordinat yang konsisten. Tahap ini juga menjadi dasar untuk memisahkan analisis per halaman sehingga hasilnya dapat disimpan dan dilacak dengan lebih mudah.

#### 5.3.1.1 Ukuran halaman dan sistem koordinat

PDF extractor membaca ukuran setiap halaman untuk mengetahui lebar, tinggi, dan sistem koordinat yang dipakai. Informasi ini penting karena seluruh bounding box harus dihitung dalam ruang koordinat yang sama. Dengan mengetahui ukuran halaman sejak awal, sistem dapat menjaga konsistensi posisi elemen dari tahap ekstraksi hingga visualisasi di frontend.

#### 5.3.1.2 Teks dan objek visual pada halaman

Selain ukuran halaman, extractor juga membaca teks, tabel, gambar, garis, dan shape yang terdapat pada halaman PDF. Data ini tidak hanya penting untuk alignment, tetapi juga untuk membedakan area teks biasa dari area visual seperti tabel atau gambar. Dengan pendekatan ini, analisis visual tidak bergantung pada satu jenis elemen saja, melainkan mencakup berbagai komponen layout dokumen.

#### 5.3.1.3 Pemisahan elemen per halaman

Extractor memisahkan hasil pembacaan berdasarkan halaman sehingga setiap unit visual memiliki konteks halaman yang jelas. Pemisahan ini penting karena proses alignment dan fusion berjalan per halaman sebelum digabungkan pada tingkat dokumen. Selain itu, penyimpanan per halaman memudahkan worker validasi dan frontend untuk menelusuri lokasi kesalahan secara lebih terarah.

### 5.3.2 Ekstraksi Karakter dan Penggabungan

Subbab ini membahas alasan sistem mengekstraksi informasi hingga tingkat karakter. Pendekatan ini dipilih untuk memperoleh koordinat yang lebih presisi daripada ekstraksi yang langsung berbentuk blok besar. Setelah itu, karakter-karakter tersebut digabungkan kembali menjadi unit yang lebih bermakna untuk dipakai pada tahap alignment.

#### 5.3.2.1 Bounding box tingkat karakter

PDF extractor mengambil koordinat hingga tingkat karakter agar sistem tidak kehilangan detail posisi teks. Setiap karakter memiliki bounding box yang dapat dipakai sebagai dasar pembentukan unit teks secara bertahap. Strategi ini membuat sistem lebih tahan terhadap kasus pemenggalan, spasi tidak seragam, atau fragmentasi teks di PDF.

#### 5.3.2.2 Penggabungan menjadi grup teks

Karakter-karakter yang berdekatan atau saling beririsan kemudian dikelompokkan menjadi unit yang lebih stabil seperti kata, baris, atau blok teks. Proses ini dilakukan dengan memperhatikan jarak spasial dan keteraturan posisi di halaman. Hasilnya adalah representasi teks yang tetap presisi secara visual tetapi lebih mudah dicocokkan dengan elemen struktur dokumen.

#### 5.3.2.3 Stabilitas koordinat hasil penggabungan

Penggabungan karakter menjadi grup teks membantu mengurangi noise koordinat yang terlalu granular. Dengan unit yang lebih stabil, proses overlap dan alignment dapat dilakukan secara lebih andal pada tahap selanjutnya. Oleh karena itu, tahap ini berperan penting dalam menjaga kualitas bounding box yang akan dipakai sebagai lokasi elemen final.

### 5.3.3 Penanganan Area Khusus

Subbab ini membahas bahwa tidak semua area dokumen dapat diproses dengan aturan yang sama. Elemen seperti tabel, gambar, dan shape memiliki pola visual yang berbeda dari paragraf biasa sehingga memerlukan penanganan khusus. Penyesuaian ini penting agar extractor tidak menghasilkan grup teks yang keliru pada area visual yang kompleks.

#### 5.3.3.1 Area tabel

Pada area tabel, teks cenderung tersebar per sel dan tidak selalu mengikuti alur baca paragraf. Karena itu, extractor memperlakukan area tabel secara lebih konservatif dengan mempertimbangkan struktur grid dan kedekatan spasial antarfragmen. Pendekatan ini membantu menjaga agar tabel tetap dikenali sebagai blok visual yang koheren, bukan terpecah menjadi paragraf acak.

#### 5.3.3.2 Gambar dan shape

Extractor juga menangkap objek visual nonteks seperti gambar dan shape yang ada pada halaman PDF. Informasi ini penting karena beberapa elemen dokumen, misalnya segmen program atau tabel yang disisipkan sebagai gambar, tidak dapat dipahami hanya dari teks hasil DOCX. Dengan memasukkan objek visual nonteks, sistem memiliki dasar yang lebih baik untuk mengenali fungsi elemen pada halaman.

#### 5.3.3.3 Output unit PDF

Keluaran akhir tahap ini adalah sekumpulan unit PDF yang telah memiliki teks, jenis objek, dan koordinat visual. Unit-unit ini menjadi dasar untuk menghitung overlap dengan prediksi Docling dan untuk dicocokkan ke elemen Open XML. Dengan demikian, output extractor berfungsi sebagai representasi visual detail yang menopang keseluruhan pipeline analisis visual.

## 5.4 Penyelarasan dengan Elemen Struktur Dokumen

Bagian ini membahas bagaimana sistem menghubungkan tampilan PDF dengan struktur logis dokumen. Penyelarasan diperlukan karena elemen yang tampil pada PDF tidak selalu langsung dapat diidentifikasi hanya dari posisi atau label visualnya. Tahap alignment membuat setiap elemen Open XML memiliki estimasi lokasi pada halaman sehingga dapat ditelusuri kembali saat validasi berlangsung.

### 5.4.1 Dasar Penyelarasan

Subbab ini menjelaskan dua representasi dokumen yang harus dipertemukan, yaitu struktur Open XML dan unit visual PDF. Struktur Open XML menyimpan urutan logis dokumen, sedangkan PDF menampilkan hasil visual akhirnya. Tanpa mekanisme penyelarasan, kedua representasi ini akan tetap terpisah dan sulit dipakai bersama pada proses validasi.

#### 5.4.1.1 Elemen Open XML sebagai representasi struktur

Elemen Open XML yang sudah diekstraksi ke database memuat jenis elemen, urutan, dan isi teks yang merepresentasikan struktur dokumen. Data ini mencerminkan bagaimana dokumen dibangun secara logis pada file DOCX. Karena itu, elemen Open XML menjadi acuan utama untuk mengetahui identitas elemen yang sebenarnya, bukan hanya penampilannya pada PDF.

#### 5.4.1.2 Unit PDF sebagai representasi tampilan

Unit PDF menggambarkan tampilan aktual dokumen pada halaman setelah melalui proses konversi. Representasi ini penting karena validasi visual harus memeriksa objek yang benar-benar muncul pada halaman, termasuk posisi, ukuran, dan relasi antarobjek. Dengan demikian, unit PDF berfungsi sebagai pasangan visual terhadap elemen logis yang berasal dari Open XML.

#### 5.4.1.3 Kebutuhan pemetaan struktur dan visual

Sistem perlu memetakan kedua representasi tersebut agar setiap elemen struktur dapat diketahui lokasi visualnya. Pemetaan ini juga memungkinkan label visual hasil Docling ditempelkan ke elemen yang benar secara logis. Oleh karena itu, alignment bukan hanya proses teknis pencocokan teks, tetapi fondasi integrasi antara struktur dan tampilan dokumen.

### 5.4.2 Metode Pencocokan Teks

Subbab ini membahas metode utama yang dipakai sistem untuk mencari pasangan antara elemen Open XML dan unit PDF. Implementasi memanfaatkan kemiripan teks sebagai sinyal utama karena banyak elemen masih dapat dikenali dari kontennya. Namun, kemiripan teks tidak digunakan sendirian, melainkan dipadukan dengan urutan dan informasi spasial.

#### 5.4.2.1 Pemanfaatan SequenceMatcher

Sistem menggunakan `difflib.SequenceMatcher` untuk menghitung tingkat kemiripan antara teks dari elemen struktur dan kandidat teks pada PDF. Metode ini cukup cocok karena mampu menangani perbedaan kecil seperti spasi, pemenggalan, atau fragmentasi teks. Dengan demikian, pencocokan tidak harus menuntut kesamaan string secara persis untuk tetap menemukan pasangan yang benar.

#### 5.4.2.2 Perhitungan kemiripan, urutan, dan posisi

Selain skor kemiripan teks, alignment juga mempertimbangkan urutan elemen dan kedekatan posisinya pada halaman. Pendekatan ini penting agar elemen yang sama-sama mirip secara teks tidak tertukar ketika muncul di bagian halaman yang berbeda. Kombinasi beberapa sinyal tersebut membuat proses alignment lebih tahan terhadap noise yang muncul pada hasil konversi PDF.

#### 5.4.2.3 Pemilihan kandidat pencocokan terbaik

Untuk setiap elemen struktur, sistem memilih kandidat PDF yang memberikan kombinasi skor paling masuk akal. Pemilihan ini tidak hanya berdasarkan skor tertinggi, tetapi juga mempertimbangkan konsistensi terhadap elemen sebelum dan sesudahnya. Dengan strategi tersebut, alignment berupaya menjaga agar hasil pemetaan tetap stabil pada tingkat dokumen, bukan hanya benar pada satu elemen terpisah.

### 5.4.3 Penyelarasan Elemen Kompleks

Subbab ini membahas bahwa beberapa elemen tidak dapat dicocokkan dengan aturan sederhana. Paragraf panjang, judul, list, dan tabel memiliki karakteristik berbeda sehingga alignment perlu memakai strategi tambahan. Kasus-kasus ini penting dibahas karena justru sering menjadi sumber kesalahan pada analisis visual dokumen.

#### 5.4.3.1 Paragraf panjang dan teks terfragmentasi

Paragraf panjang sering terpecah menjadi beberapa unit kecil pada PDF sehingga pencocokannya tidak cukup dilakukan sekali. Sistem mengatasi hal ini dengan mencocokkan bagian teks secara bertahap dan kemudian menggabungkan hasilnya menjadi satu estimasi lokasi. Strategi tersebut membuat paragraf tetap dapat dipetakan walaupun tampil terfragmentasi pada halaman PDF.

#### 5.4.3.2 Judul, list, dan pola khusus

Untuk judul, alignment memanfaatkan pola penomoran dan kecenderungan posisi di bagian tertentu halaman. Untuk list, sistem memperhatikan token penanda seperti bullet, nomor, dan pola indentasi visual yang muncul pada hasil extractor. Pendekatan berbasis pola ini membantu sistem mengenali fungsi elemen yang tidak selalu dapat dijelaskan hanya oleh kemiripan teks.

#### 5.4.3.3 Hasil alignment dan ketidakpastian

Keluaran alignment berupa pasangan antara elemen struktur dan unit PDF beserta estimasi bounding box elemen tersebut. Jika kecocokan tidak cukup kuat, elemen dapat ditandai sebagai hasil yang belum pasti dan ditangani hati-hati pada tahap fusion. Dengan mekanisme ini, sistem tidak memaksakan pemetaan yang meragukan menjadi hasil final tanpa pemeriksaan tambahan.

## 5.5 Fusion dan Pemanfaatan Hasil Visual

Bagian ini membahas penggabungan antara label visual dari Docling dan hasil alignment dari extractor serta struktur dokumen. Tahap fusion adalah inti pembentukan elemen visual final karena di sinilah label, posisi, dan konteks struktur diputuskan bersama. Setelah fusion selesai, hasil visual dapat disimpan ke database dan dipakai oleh worker validasi maupun frontend.

### 5.5.1 Penggabungan Docling dan Alignment

Subbab ini menjelaskan bagaimana dua sumber informasi utama digabungkan dalam satu representasi elemen. Alignment memberikan hubungan ke elemen dokumen dan estimasi posisi yang lebih terikat pada struktur, sedangkan Docling memberi konteks layout visual. Penggabungan keduanya membuat sistem tidak hanya tahu di mana elemen berada, tetapi juga apa fungsi visualnya pada halaman.

#### 5.5.1.1 Pengukuran overlap

Fusion membandingkan bounding box hasil alignment dengan bounding box hasil Docling untuk mencari pasangan yang paling sesuai. Rasio overlap dipakai sebagai dasar utama untuk menilai kedekatan dua objek visual tersebut. Pendekatan ini sesuai dengan sifat data yang sama-sama berada pada ruang koordinat halaman PDF.

#### 5.5.1.2 Penetapan label akhir

Setelah pasangan ditemukan, label visual dari Docling ditempelkan ke elemen yang telah dipetakan melalui alignment. Pada tahap ini sistem tetap dapat melakukan koreksi jika label mentah tidak sesuai dengan konteks struktur atau pola teks elemen. Dengan demikian, label akhir merupakan hasil keputusan gabungan, bukan salinan langsung dari keluaran model.

#### 5.5.1.3 Pembentukan elemen visual final

Elemen visual final memuat teks ringkas, halaman, koordinat bounding box, label visual, dan keterkaitan ke elemen dokumen. Bentuk hasil ini lebih berguna daripada prediksi mentah karena sudah dapat langsung dipakai pada tahap validasi. Oleh sebab itu, fusion dapat dipandang sebagai titik perubahan dari data analisis menjadi data operasional sistem.

### 5.5.2 Koreksi Kasus Khusus

Subbab ini membahas koreksi tambahan yang diperlukan agar hasil fusion tidak berhenti pada pencocokan umum saja. Dokumen tugas akhir memiliki banyak elemen yang secara visual tampak mirip, tetapi fungsi strukturnya berbeda. Karena itu, sistem menerapkan aturan tambahan untuk beberapa jenis elemen yang sering menimbulkan ambiguitas.

#### 5.5.2.1 Header dan footer

Sistem menggunakan informasi margin dan zona halaman untuk mengoreksi kemungkinan salah klasifikasi pada area header dan footer. Koreksi ini penting karena teks yang dekat dengan tepi halaman dapat salah dianggap sebagai bagian isi utama atau sebaliknya. Dengan pembatasan zona tersebut, hasil visual menjadi lebih konsisten terhadap struktur halaman dokumen.

#### 5.5.2.2 Caption dan footnote

Untuk caption, fusion memeriksa kedekatan spasial dengan gambar atau objek visual terkait agar pasangan caption dan objek tidak tertukar. Untuk footnote, sistem memanfaatkan posisi di bagian bawah halaman dan, pada dokumen tertentu, mengaitkannya dengan data note hasil ekstraksi Open XML. Pendekatan ini membuat elemen yang secara tekstual mirip paragraf biasa tetap dapat dibedakan berdasarkan fungsi visual dan strukturnya.

#### 5.5.2.3 Label struktural tambahan

Selain label visual umum, sistem juga menurunkan label struktural seperti judul bab atau judul subbab berdasarkan pola teks dan konteks alignment. Penurunan label ini penting karena tidak semua kebutuhan validasi dapat dipenuhi hanya dengan kategori visual seperti `text` atau `picture`. Dengan label tambahan tersebut, worker validasi dapat menerapkan aturan yang lebih spesifik sesuai pedoman penulisan.

### 5.5.3 Integrasi ke Sistem

Subbab ini menjelaskan bagaimana hasil fusion dimasukkan kembali ke dalam pipeline sistem. Penyimpanan hasil visual bukan hanya langkah teknis, tetapi juga penanda bahwa dokumen siap memasuki tahap validasi. Selain itu, hasil yang tersimpan ini menjadi dasar utama untuk visualisasi kesalahan pada antarmuka pengguna.

#### 5.5.3.1 Penyimpanan ke tabel elemen visual

Hasil fusion disimpan ke tabel `dokumen_elemen_visual` agar setiap elemen visual dapat diakses kembali secara persisten. Penyimpanan ini dilakukan per halaman dan dikaitkan dengan elemen dokumen yang relevan. Dengan penyimpanan terstruktur tersebut, worker lain tidak perlu mengulang proses visual dari awal untuk membaca hasil analisis.

#### 5.5.3.2 Pembaruan status dan handoff

Setelah penyimpanan berhasil, worker visual memperbarui `antrian_labeling_status` menjadi `completed`. Pada saat yang sama, `antrian_validation_status` diubah menjadi `in_queue` agar validation worker dapat melanjutkan proses dokumen yang sama. Mekanisme ini memastikan handoff antartahap berlangsung eksplisit, terdokumentasi, dan mudah ditelusuri.

#### 5.5.3.3 Penggunaan pada validasi dan frontend

Worker validasi memakai hasil visual ini untuk menentukan lokasi kesalahan dan memahami jenis elemen yang sedang diperiksa. Frontend memanfaatkan data yang sama untuk menampilkan preview halaman dan penandaan lokasi error secara lebih jelas kepada pengguna. Dengan kata lain, satu hasil fusion melayani dua kebutuhan utama sekaligus, yaitu validasi internal dan penyajian hasil ke pengguna.

## 5.6 Evaluasi Akurasi

Bagian ini membahas bagaimana kualitas analisis visual dinilai secara sistematis. Karena layanan visual memengaruhi validasi dan visualisasi kesalahan, akurasinya perlu dianalisis pada tingkat klasifikasi dan lokasi. Evaluasi juga penting untuk menjelaskan secara ilmiah alasan peralihan dari LayoutLMv3 ke Docling pada implementasi akhir.

### 5.6.1 Rancangan Evaluasi

Subbab ini menjelaskan cara menyusun evaluasi yang relevan dengan sistem yang dibangun. Evaluasi tidak cukup hanya mengukur apakah label benar, tetapi juga harus menilai apakah lokasi elemen cukup tepat untuk dipakai validasi. Oleh karena itu, rancangan evaluasi perlu memadukan acuan kategorisasi elemen dan acuan lokasi pada halaman.

#### 5.6.1.1 Data uji dan label acuan

Data uji sebaiknya diambil dari dokumen tugas akhir yang merepresentasikan variasi struktur nyata pada sistem, seperti judul, tabel, gambar, list, dan footnote. Setiap elemen penting diberi label acuan agar hasil Docling dan hasil fusion dapat dibandingkan terhadap standar yang sama. Dengan acuan tersebut, evaluasi tidak hanya bersifat impresi visual, tetapi memiliki dasar pembandingan yang eksplisit.

#### 5.6.1.2 Metrik kategori

Untuk menilai ketepatan label, metrik seperti precision, recall, dan F1-score dapat digunakan pada tiap kategori elemen. Metrik ini membantu menunjukkan apakah model cenderung melewatkan elemen tertentu atau justru terlalu sering memberi label yang salah. Pemakaian metrik kategori sesuai dengan fungsi utama Docling sebagai pemberi label visual awal pada sistem.

#### 5.6.1.3 Metrik lokasi dan konsistensi

Selain kategori, evaluasi perlu menilai ketepatan lokasi bounding box, misalnya melalui overlap atau IoU terhadap anotasi acuan. Konsistensi antarhalaman juga penting karena sistem memproses dokumen panjang yang mengandung elemen serupa berulang kali. Dengan memasukkan aspek lokasi dan konsistensi, evaluasi menjadi lebih selaras dengan kebutuhan validasi aktual pada sistem.

### 5.6.2 Analisis Akurasi Hasil Docling

Subbab ini membahas akurasi hasil Docling berdasarkan kelompok elemen yang paling relevan pada dokumen tugas akhir. Pemisahan analisis per kelompok diperlukan karena tingkat kesulitan setiap kategori tidak sama. Dengan analisis ini, pembahasan akurasi menjadi lebih tajam dan tidak berhenti pada satu angka agregat saja.

#### 5.6.2.1 Akurasi elemen teks dan heading

Elemen teks dan heading umumnya memiliki jumlah paling besar dan menjadi tulang punggung struktur dokumen. Akurasi pada kategori ini penting karena kesalahan pada heading dapat berpengaruh langsung pada validasi struktur bab dan subbab. Analisis pada bagian ini dapat menunjukkan apakah model cukup baik membedakan paragraf biasa dari elemen judul yang memiliki fungsi struktural lebih tinggi.

#### 5.6.2.2 Akurasi elemen tabel, gambar, dan caption

Kategori tabel, gambar, dan caption penting karena sering kali tidak dapat dikenali hanya dari struktur DOCX, terutama jika disisipkan sebagai objek visual. Akurasi pada kategori ini menentukan seberapa baik sistem dapat memisahkan konten utama dari elemen pendukung yang memiliki aturan format berbeda. Hasil analisis pada bagian ini juga berkaitan langsung dengan kualitas fusion pada area visual kompleks.

#### 5.6.2.3 Akurasi elemen list dan footnote

List dan footnote merupakan elemen yang cenderung menimbulkan ambiguitas karena tampilannya bisa mendekati paragraf biasa. Akurasi pada kategori ini penting untuk menilai apakah sistem benar-benar terbantu oleh Docling dan aturan koreksi tambahan. Bagian ini dapat menunjukkan seberapa jauh pendekatan visual mampu memperbaiki keterbatasan yang tidak mudah diatasi hanya dengan ekstraksi struktur.

### 5.6.3 Perbandingan dan Analisis Kesalahan

Subbab ini menempatkan hasil evaluasi dalam konteks keputusan desain sistem. Perbandingan dengan LayoutLMv3 diperlukan untuk menjelaskan alasan peralihan model secara lebih ilmiah. Analisis kesalahan juga penting agar pembahasan tidak hanya menonjolkan keberhasilan, tetapi juga menunjukkan batas kemampuan sistem yang sebenarnya.

#### 5.6.3.1 Perbandingan LayoutLMv3 dan Docling

Perbandingan dilakukan pada kategori yang sama agar terlihat model mana yang lebih stabil untuk dokumen tugas akhir. Fokus pembahasan bukan hanya pada angka akurasi tertinggi, tetapi juga pada kemudahan integrasi dan konsistensi hasil dalam pipeline nyata. Jika Docling memberi hasil yang lebih dapat diandalkan walaupun bukan selalu yang paling kompleks secara model, maka keputusan implementasi menjadi lebih dapat dipertanggungjawabkan.

#### 5.6.3.2 Sumber kesalahan dominan

Kesalahan dominan biasanya muncul pada dokumen dengan layout padat, elemen visual kecil, atau objek campuran seperti tabel yang dirender sebagai gambar. Sumber kesalahan lain dapat berasal dari fragmentasi teks PDF, bounding box yang overlap, dan alignment yang belum cukup kuat pada elemen panjang. Identifikasi sumber kesalahan ini penting untuk menunjukkan bahwa keterbatasan sistem tidak hanya berasal dari model AI, tetapi juga dari karakter dokumen dan pipeline pemrosesannya.

#### 5.6.3.3 Implikasi pengembangan lanjutan

Hasil evaluasi dapat digunakan untuk menentukan arah pengembangan berikutnya, misalnya memperkuat aturan fusion atau menambah evaluasi bbox yang lebih detail. Pengembangan juga dapat diarahkan pada kategori elemen yang akurasinya paling lemah agar perbaikan lebih terukur. Dengan demikian, evaluasi akurasi tidak berhenti sebagai hasil pengujian, tetapi menjadi dasar perencanaan peningkatan sistem pada penelitian lanjutan.
