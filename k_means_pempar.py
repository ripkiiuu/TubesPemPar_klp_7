"""
k_means_pempar_simple.py
========================
K-Means Paralel dengan MPI — versi ringkas & mudah dipahami

CARA PAKAI:
  mpiexec -n 2 python k_means_pempar_simple.py --mode batch --output-csv hasil.csv --quiet
  mpiexec -n 4 python k_means_pempar_simple.py --mode batch --output-csv hasil.csv --quiet
  mpiexec -n 6 python k_means_pempar_simple.py --mode batch --output-csv hasil.csv --quiet
  mpiexec -n 8 python k_means_pempar_simple.py --mode batch --output-csv hasil.csv --quiet

  lalu:
  python buat_grafik.py --csv hasil.csv
"""

from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import argparse
import os

# ============================================================
# INISIALISASI MPI
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # ID proses ini (0 = bos, sisanya pekerja)
size = comm.Get_size()   # total proses yang berjalan


# ============================================================
# FUNGSI BANTU
# ============================================================

def load_data(csv_path, n_data):
    """Baca CSV, ambil kolom Quantity & UnitPrice, buang outlier."""
    df = pd.read_csv(csv_path, encoding='unicode_escape')
    df = df[['Quantity', 'UnitPrice']].dropna()
    df = df[
        (df['Quantity'] > 0) & (df['Quantity'] < 50) &
        (df['UnitPrice'] > 0) & (df['UnitPrice'] < 20)
    ]
    return df.values.astype(np.float64)[:n_data]


def hitung_jarak(data, centroids):
    """
    Hitung jarak Euclidean setiap titik ke setiap centroid.
    Hasilnya: matrix (jumlah_data x K)
    """
    # data[:, np.newaxis] ubah shape jadi (N, 1, 2)
    # centroids shape (K, 2)
    # hasilnya broadcast jadi (N, K, 2) → norm → (N, K)
    return np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)


def simpan_csv(output_csv, n_data, n_procs, t_serial, t_paralel):
    """Append satu baris hasil ke file CSV."""
    row = {
        'n_data':        n_data,
        'K':             3,
        'n_procs':       n_procs,
        'mode':          'both',
        'serial_total':  t_serial,
        'serial_komp':   t_serial,
        'paralel_total': t_paralel,
        'paralel_komp':  t_paralel,
        'speedup_total': t_serial / t_paralel if t_paralel > 0 else 0,
        'speedup_komp':  '',
        'eff_total':     '',
        'eff_komp':      '',
    }
    file_ada = os.path.exists(output_csv)
    pd.DataFrame([row]).to_csv(output_csv, mode='a', header=not file_ada, index=False)


# ============================================================
# K-MEANS SERIAL
# ============================================================

def kmeans_serial(data, K=3, maks_iterasi=100, tol=1e-4, seed=42):
    """
    K-Means biasa (1 prosesor).
    Dijalankan hanya oleh rank 0 sebagai baseline perbandingan.

    Alur:
      1. Pilih K titik awal sebagai centroid
      2. Assign tiap titik ke centroid terdekat
      3. Update centroid = rata-rata titik di cluster itu
      4. Ulangi sampai centroid tidak bergerak (konvergen)
    """
    rng = np.random.RandomState(seed)
    # Pilih K titik acak sebagai centroid awal
    centroids = data[rng.choice(len(data), K, replace=False)].copy()

    mulai = time.time()

    for _ in range(maks_iterasi):
        # Langkah 2: assign
        labels = np.argmin(hitung_jarak(data, centroids), axis=1)

        # Langkah 3: update centroid
        centroids_baru = np.array([
            data[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(K)
        ])

        # Langkah 4: cek konvergensi
        if np.linalg.norm(centroids_baru - centroids) < tol:
            break
        centroids = centroids_baru

    return time.time() - mulai   # kembalikan waktu eksekusi


# ============================================================
# K-MEANS PARALEL
# ============================================================

def kmeans_paralel(data_all, K=3, maks_iterasi=100, tol=1e-4, seed=42):
    """
    K-Means paralel menggunakan MPI.

    Alur:
      Rank 0  → potong data, kirim ke semua proses, broadcast centroid awal
      Semua   → hitung jarak & assign di data lokal masing-masing
      Semua   → allreduce: jumlahkan sum & count dari semua proses
      Semua   → update centroid (sama di semua proses)
      Rank 0  → kumpulkan hasil akhir
    """

    # --- RANK 0: siapkan data & potong ---
    if rank == 0:
        rng = np.random.RandomState(seed)
        centroids = data_all[rng.choice(len(data_all), K, replace=False)].copy()
        potongan = np.array_split(data_all, size)  # bagi jadi `size` bagian
    else:
        centroids = None
        potongan  = None

    mulai = time.time()

    # --- DISTRIBUSI: kirim 1 potongan ke tiap proses ---
    data_lokal = comm.scatter(potongan, root=0)    # tiap proses dapat bagiannya
    centroids  = comm.bcast(centroids, root=0)     # semua proses dapat centroid awal

    # --- ITERASI K-MEANS ---
    for _ in range(maks_iterasi):

        # Tiap proses assign titik lokalnya ke centroid terdekat
        labels = np.argmin(hitung_jarak(data_lokal, centroids), axis=1)

        # Tiap proses hitung sum & count lokal
        local_sums   = np.array([
            data_lokal[labels == k].sum(axis=0) if np.any(labels == k) else np.zeros(2)
            for k in range(K)
        ])
        local_counts = np.array([
            np.sum(labels == k) for k in range(K)
        ], dtype=np.float64)

        # Kumpulkan dari semua proses → jumlahkan (allreduce)
        # allreduce: tiap proses kirim, semua dijumlah, hasil dikirim balik ke semua
        total_sums   = comm.allreduce(local_sums,   op=MPI.SUM)
        total_counts = comm.allreduce(local_counts, op=MPI.SUM)

        # Update centroid (dihitung sama di semua proses)
        centroids_baru = np.array([
            total_sums[k] / total_counts[k] if total_counts[k] > 0 else centroids[k]
            for k in range(K)
        ])

        # Cek konvergensi — tidak perlu allreduce lagi karena centroid sama di semua proses
        if np.linalg.norm(centroids_baru - centroids) < tol:
            break
        centroids = centroids_baru

    waktu = time.time() - mulai
    return waktu   # semua proses return waktu, tapi yang dipakai hanya rank 0


# ============================================================
# JALANKAN SATU EKSPERIMEN (serial + paralel, beberapa trial)
# ============================================================

def jalankan_eksperimen(csv_path, n_data, K, n_trials, output_csv, quiet):
    """
    Jalankan serial dan paralel masing-masing n_trials kali,
    ambil rata-rata waktunya, simpan ke CSV.
    """

    # Rank 0 load data (proses lain tidak perlu)
    if rank == 0:
        data = load_data(csv_path, n_data)
        if not quiet:
            print(f"  Data: {len(data)} baris")
    else:
        data = None

    # --- SERIAL (hanya rank 0) ---
    waktu_serial_list = []
    if rank == 0:
        for trial in range(n_trials):
            t = kmeans_serial(data, K=K, seed=42 + trial)
            waktu_serial_list.append(t)
        rata_serial = np.mean(waktu_serial_list)
        if not quiet:
            print(f"  Serial rata-rata : {rata_serial:.4f} detik")
    else:
        rata_serial = None

    # --- PARALEL (semua proses ikut) ---
    waktu_paralel_list = []
    for trial in range(n_trials):
        # Rank 0 broadcast data ke semua proses tiap trial
        data_broadcast = comm.bcast(data, root=0)
        t = kmeans_paralel(data_broadcast, K=K, seed=42 + trial)
        if rank == 0:
            waktu_paralel_list.append(t)

    if rank == 0:
        rata_paralel = np.mean(waktu_paralel_list)
        if not quiet:
            print(f"  Paralel rata-rata: {rata_paralel:.4f} detik ({size} proses)")
            print(f"  Speedup          : {rata_serial/rata_paralel:.2f}x")

        # Simpan ke CSV
        if output_csv:
            simpan_csv(output_csv, n_data, size, rata_serial, rata_paralel)


# ============================================================
# BATCH: jalankan semua ukuran data
# ============================================================

def jalankan_batch(csv_path, K, n_trials, output_csv, quiet):
    daftar_n_data = [1000, 5000, 10000, 25000, 50000, 75000, 100000]

    for n_data in daftar_n_data:
        if rank == 0 and not quiet:
            print(f"\n[{n_data:,} data | {size} proses]")

        jalankan_eksperimen(csv_path, n_data, K, n_trials, output_csv, quiet)

    if rank == 0:
        print(f"\nSelesai. Hasil disimpan di: {output_csv}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        type=str,   default='data.csv')
    parser.add_argument('--K',          type=int,   default=3)
    parser.add_argument('--trials',     type=int,   default=3)
    parser.add_argument('--output-csv', type=str,   default='')
    parser.add_argument('--mode',       type=str,   default='batch')
    parser.add_argument('--quiet',      action='store_true')
    parser.add_argument('--no-plot',    action='store_true')  # diabaikan, grafik dibuat lewat buat_grafik.py
    args = parser.parse_args()

    jalankan_batch(args.csv, args.K, args.trials, args.output_csv, args.quiet)


if __name__ == '__main__':
    main()