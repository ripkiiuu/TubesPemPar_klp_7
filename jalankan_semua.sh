#!/bin/bash
# =============================================================
# jalankan_semua.sh
# =============================================================
# Script ini menjalankan k_means_pempar.py untuk SEMUA kombinasi
# jumlah proses (2, 4, 6, 8), lalu membuat kedua grafik otomatis.
#
# CARA PAKAI:
#   chmod +x jalankan_semua.sh
#   ./jalankan_semua.sh
#
# SYARAT:
#   - mpi4py sudah terinstall  (pip install mpi4py)
#   - file data.csv ada di folder yang sama
#   - file k_means_pempar.py ada di folder yang sama
# =============================================================

CSV_DATA="data.csv"
OUTPUT_CSV="hasil.csv"
KODE="k_means_pempar.py"
TRIALS=3         # tiap konfigurasi diulang 3x lalu diambil rata-rata
K=3              # jumlah cluster

# Ukuran data yang diuji (sesuai grafik di paper)
DAFTAR_DATA="1000,5000,10000,25000,50000,75000,100000"

# Hapus hasil lama supaya tidak tercampur
if [ -f "$OUTPUT_CSV" ]; then
    echo "[INFO] Menghapus file hasil lama: $OUTPUT_CSV"
    rm "$OUTPUT_CSV"
fi

echo "=============================================="
echo " MEMULAI EKSPERIMEN K-MEANS PARALEL"
echo "=============================================="

# Loop untuk tiap jumlah proses
for N_PROCS in 2 4 6 8; do
    echo ""
    echo "----------------------------------------------"
    echo " Menjalankan dengan $N_PROCS proses..."
    echo "----------------------------------------------"

    mpirun -n $N_PROCS python $KODE \
        --mode batch \
        --csv $CSV_DATA \
        --K $K \
        --trials $TRIALS \
        --output-csv $OUTPUT_CSV \
        --no-plot \
        --quiet

    if [ $? -ne 0 ]; then
        echo "[ERROR] Gagal saat menjalankan dengan $N_PROCS proses. Berhenti."
        exit 1
    fi

    echo "[OK] Selesai: $N_PROCS proses"
done

echo ""
echo "=============================================="
echo " SEMUA EKSPERIMEN SELESAI"
echo " Hasil disimpan di: $OUTPUT_CSV"
echo "=============================================="

# Buat kedua grafik otomatis
echo ""
echo "[INFO] Membuat grafik..."
python buat_grafik.py --csv $OUTPUT_CSV

echo ""
echo "=============================================="
echo " SELESAI. Cek file:"
echo "   - grafik_pengujian1.png"
echo "   - grafik_pengujian2.png"
echo "=============================================="