from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import argparse
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()   
size = comm.Get_size()   

def load_data(csv_path, n_data):
    df = pd.read_csv(csv_path, encoding='unicode_escape')
    df = df[['Quantity', 'UnitPrice']].dropna()
    df = df[
        (df['Quantity'] > 0) & (df['Quantity'] < 50) &
        (df['UnitPrice'] > 0) & (df['UnitPrice'] < 20)
    ]
    return df.values.astype(np.float64)[:n_data]


def hitung_jarak(data, centroids):
    
    return np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)


def simpan_csv(output_csv, n_data, K, n_procs, t_serial, t_paralel):
    row = {
        'n_data':        n_data,
        'K':             K,
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


def kmeans_serial(data, K=3, maks_iterasi=100, tol=1e-4, seed=42):
    rng = np.random.RandomState(seed)
    centroids = data[rng.choice(len(data), K, replace=False)].copy()

    mulai = time.time()
    iterasi = 0
    for _ in range(maks_iterasi):
        iterasi += 1
        labels = np.argmin(hitung_jarak(data, centroids), axis=1)

        centroids_baru = np.array([
            data[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(K)
        ])

        if np.linalg.norm(centroids_baru - centroids) < tol:
            break
        centroids = centroids_baru

    return time.time() - mulai, iterasi

def kmeans_paralel(data_all, K, maks_iterasi=100, tol=1e-4, seed=42):

    if rank == 0:
        rng = np.random.RandomState(seed)
        centroids = data_all[rng.choice(len(data_all), K, replace=False)].copy()
        potongan = np.array_split(data_all, size)
    else:
        centroids = None
        potongan  = None

    mulai = time.time()

    data_lokal = comm.scatter(potongan, root=0)
    centroids  = comm.bcast(centroids, root=0)

    iterasi = 0
    t_komputasi = 0.0
    for _ in range(maks_iterasi):
        iterasi += 1
        labels = np.argmin(hitung_jarak(data_lokal, centroids), axis=1)

        local_sums   = np.array([
            data_lokal[labels == k].sum(axis=0) if np.any(labels == k) else np.zeros(2)
            for k in range(K)
        ])
        local_counts = np.array([
            np.sum(labels == k) for k in range(K)
        ], dtype=np.float64)

        t0 = time.time()
        total_sums   = comm.allreduce(local_sums,   op=MPI.SUM)
        total_counts = comm.allreduce(local_counts, op=MPI.SUM)
        t_komputasi += time.time() - t0

        centroids_baru = np.array([
            total_sums[k] / total_counts[k] if total_counts[k] > 0 else centroids[k]
            for k in range(K)
        ])

        if np.linalg.norm(centroids_baru - centroids) < tol:
            break
        centroids = centroids_baru

    waktu = time.time() - mulai
    return waktu, t_komputasi, iterasi

def jalankan_eksperimen(csv_path, n_data, K, n_trials, output_csv):

    if rank == 0:
        data = load_data(csv_path, n_data)
    else:
        data = None

    # ── Serial (hanya rank 0) ───────────────────────────
    rata_serial = None
    rata_km_serial = None
    if rank == 0:
        t_km_list, t_total_list = [], []
        for trial in range(n_trials):
            t0 = time.time()
            data_trial = load_data(csv_path, n_data)
            t_prep = time.time() - t0
            t_km, _ = kmeans_serial(data_trial, K=K, seed=42 + trial)
            t_km_list.append(t_km)
            t_total_list.append(t_prep + t_km)
        rata_serial = np.mean(t_total_list)
        rata_km_serial = np.mean(t_km_list)

    # ── Paralel (semua proses) ──────────────────────────
    t_total_list, t_komputasi_list = [], []
    for trial in range(n_trials):
        data_bc = comm.bcast(data, root=0)
        t_total, t_komputasi, _ = kmeans_paralel(data_bc, K=K, seed=42 + trial)
        if rank == 0:
            t_total_list.append(t_total)
            t_komputasi_list.append(t_komputasi)

    if rank == 0:
        rata_paralel = np.mean(t_total_list)
        rata_komputasi = np.mean(t_komputasi_list)
        if output_csv:
            simpan_csv(output_csv, n_data, K, size, rata_serial, rata_paralel)
        return rata_serial, rata_km_serial, rata_paralel, rata_komputasi
    return None, None, None, None

def jalankan_batch(csv_path, K, n_trials, output_csv, quiet):
    daftar_n_data = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
    hasil = []

    for n_data in daftar_n_data:
        row = jalankan_eksperimen(csv_path, n_data, K, n_trials, output_csv)
        if rank == 0:
            hasil.append((n_data, *row))

    if rank == 0 and not quiet:
        print(f"\n{'='*72}")
        print(f"  RINGKASAN  |  K={K}  |  {size} Proses")
        print(f"{'='*72}")
        print(f"  {'N Data':>8} | {'Serial(s)':>10} | {'Paralel(s)':>10} | {'Speedup':>7} | {'Kmeans(s)':>9} | {'KomputasiPar(s)':>15}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}-+-{'-'*15}")
        for n_data, t_ser, t_km, t_par, t_komp in hasil:
            speedup = t_ser / t_par if t_par > 0 else 0
            print(f"  {n_data:>8,} | {t_ser:>10.4f} | {t_par:>10.4f} | {speedup:>6.2f}x | {t_km:>9.4f} | {t_komp:>15.4f}")
        print(f"{'='*72}")

    if rank == 0:
        print(f"\nSelesai. Hasil disimpan di: {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        type=str,   default='data.csv')
    parser.add_argument('--K',          type=int,   default=3)
    parser.add_argument('--trials',     type=int,   default=3)
    parser.add_argument('--output-csv', type=str,   default='')
    parser.add_argument('--mode',       type=str,   default='batch')
    parser.add_argument('--quiet',      action='store_true')
    parser.add_argument('--no-plot',    action='store_true')
    args = parser.parse_args()

    jalankan_batch(args.csv, args.K, args.trials, args.output_csv, args.quiet)


if __name__ == '__main__':
    main()