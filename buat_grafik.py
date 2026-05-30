"""
buat_grafik.py
==============
Script untuk membuat Grafik Pengujian 1 & 2 dari file hasil.csv

CARA PAKAI:
  python buat_grafik.py --csv hasil.csv

OUTPUT:
  grafik_pengujian1.png  → Waktu vs Jumlah Data (multi-line per proses)
  grafik_pengujian2.png  → Waktu vs Jumlah Proses (multi-line per ukuran data)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys
import os

# ============================================================
# KONFIGURASI TAMPILAN GRAFIK
# ============================================================

# Warna untuk tiap jumlah proses (sesuaikan dengan grafik di PDF)
WARNA_PROSES = {
    1: ('#d62728', '*', '1 Proses (Serial)'),   # merah
    2: ('#1f77b4', 'o', '2 Proses'),             # biru
    4: ('#2ca02c', 's', '4 Proses'),             # hijau
    6: ('#ff7f0e', 'D', '6 Proses'),             # oranye
    8: ('#9467bd', '^', '8 Proses'),             # ungu
}

# Warna untuk tiap ukuran data (Pengujian 2)
WARNA_DATA = [
    '#9467bd', '#8c564b', '#2ca02c', '#ff7f0e',
    '#1f77b4', '#d62728', '#17becf'
]


# ============================================================
# FUNGSI UTAMA
# ============================================================

def load_hasil(csv_path):
    """
    Membaca file hasil.csv dan mengembalikan DataFrame.
    
    Kolom yang dibutuhkan:
      n_data, n_procs, serial_total, paralel_total
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File '{csv_path}' tidak ditemukan.")
        print("Pastikan kamu sudah menjalankan k_means_pempar.py dengan --output-csv hasil.csv")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Cek kolom yang dibutuhkan
    kolom_wajib = ['n_data', 'n_procs', 'serial_total', 'paralel_total']
    for kol in kolom_wajib:
        if kol not in df.columns:
            print(f"[ERROR] Kolom '{kol}' tidak ada di file CSV.")
            print(f"Kolom yang ada: {list(df.columns)}")
            sys.exit(1)

    # Ambil rata-rata jika ada duplikat (n_data, n_procs)
    df = df.groupby(['n_data', 'n_procs'], as_index=False).mean(numeric_only=True)
    df = df.sort_values(['n_procs', 'n_data'])

    print(f"Data berhasil dibaca: {len(df)} baris")
    print(f"Jumlah proses yang ada: {sorted(df['n_procs'].unique())}")
    print(f"Ukuran data yang ada:   {sorted(df['n_data'].unique())}")
    return df


def buat_pengujian1(df, output='grafik_pengujian1.png'):
    """
    PENGUJIAN 1: Sumbu X = Jumlah Data, tiap garis = jumlah proses.
    
    Ini menjawab pertanyaan:
    'Semakin banyak data, waktu naik seberapa cepat untuk tiap konfigurasi proses?'
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    daftar_proses = sorted(df['n_procs'].unique())

    for n_procs in daftar_proses:
        subset = df[df['n_procs'] == n_procs].sort_values('n_data')

        if n_procs == 1:
            # Serial: pakai kolom serial_total
            waktu = subset['serial_total'].values
        else:
            # Paralel: pakai kolom paralel_total
            waktu = subset['paralel_total'].values

        n_datas = subset['n_data'].values

        # Ambil konfigurasi warna & marker
        if n_procs in WARNA_PROSES:
            warna, marker, label = WARNA_PROSES[n_procs]
        else:
            warna, marker, label = '#333333', 'x', f'{n_procs} Proses'

        ax.plot(n_datas, waktu,
                color=warna, marker=marker, label=label,
                linewidth=1.8, markersize=7, markerfacecolor='white',
                markeredgewidth=1.5)

        # Tambahkan label angka di tiap titik
        for x, y in zip(n_datas, waktu):
            ax.annotate(f'{y:.4f}',
                        xy=(x, y),
                        xytext=(0, 7),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=6.5, color=warna)

    ax.set_title('Pengujian 1: Waktu Komputasi untuk Berbagai Jumlah Data\nterhadap Berbagai Jumlah N',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Jumlah Data', fontsize=11)
    ax.set_ylabel('Waktu Komputasi (detik)', fontsize=11)

    # Format sumbu X agar tampil dengan koma (1,000 bukan 1000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xticks(sorted(df['n_data'].unique()))
    plt.xticks(rotation=30)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafik Pengujian 1 disimpan: {output}")


def buat_pengujian2(df, output='grafik_pengujian2.png'):
    """
    PENGUJIAN 2: Sumbu X = Jumlah Proses, tiap garis = ukuran data.
    
    Ini menjawab pertanyaan:
    'Semakin banyak proses, waktu turun seberapa cepat untuk tiap ukuran data?'
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    daftar_data = sorted(df['n_data'].unique())

    for i, n_data in enumerate(daftar_data):
        subset = df[df['n_data'] == n_data].sort_values('n_procs')

        # Untuk tiap ukuran data, kita gabungkan serial (n_procs=1) + paralel
        proses_list = []
        waktu_list = []

        for _, baris in subset.iterrows():
            n_procs = int(baris['n_procs'])
            proses_list.append(n_procs)

            if n_procs == 1:
                waktu_list.append(baris['serial_total'])
            else:
                waktu_list.append(baris['paralel_total'])

        warna = WARNA_DATA[i % len(WARNA_DATA)]
        label = f'{n_data:,} Data'

        ax.plot(proses_list, waktu_list,
                color=warna, marker='o', label=label,
                linewidth=1.8, markersize=7, markerfacecolor='white',
                markeredgewidth=1.5)

        # Label angka di tiap titik
        for x, y in zip(proses_list, waktu_list):
            ax.annotate(f'{y:.4f}',
                        xy=(x, y),
                        xytext=(0, 7),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=6.5, color=warna)

    ax.set_title('Pengujian 2: Waktu Komputasi untuk Berbagai Jumlah N\nterhadap Berbagai Jumlah Data',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Jumlah N (Proses)', fontsize=11)
    ax.set_ylabel('Waktu Komputasi (detik)', fontsize=11)

    # Sumbu X: hanya tampilkan nilai proses yang ada
    daftar_proses = sorted(df['n_procs'].unique())
    ax.set_xticks(daftar_proses)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafik Pengujian 2 disimpan: {output}")


def cetak_tabel_ringkasan(df):
    """
    Print tabel ringkasan waktu di terminal — berguna buat isi tabel di paper.
    """
    print("\n" + "=" * 70)
    print("TABEL RINGKASAN WAKTU KOMPUTASI (detik)")
    print("=" * 70)

    daftar_data = sorted(df['n_data'].unique())
    daftar_proses = sorted(df['n_procs'].unique())

    # Header
    header = f"{'Data':<12}"
    for p in daftar_proses:
        label = f"N={p}" if p > 1 else "Serial"
        header += f"{label:>12}"
    print(header)
    print("-" * 70)

    for n_data in daftar_data:
        row = f"{n_data:<12,}"
        for n_procs in daftar_proses:
            subset = df[(df['n_data'] == n_data) & (df['n_procs'] == n_procs)]
            if len(subset) == 0:
                row += f"{'N/A':>12}"
            else:
                if n_procs == 1:
                    waktu = subset['serial_total'].values[0]
                else:
                    waktu = subset['paralel_total'].values[0]
                row += f"{waktu:>12.4f}"
        print(row)

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Buat grafik Pengujian 1 & 2 dari hasil CSV')
    parser.add_argument('--csv', type=str, default='hasil.csv',
                        help='Path ke file hasil CSV (default: hasil.csv)')
    parser.add_argument('--out1', type=str, default='grafik_pengujian1.png',
                        help='Nama output Pengujian 1 (default: grafik_pengujian1.png)')
    parser.add_argument('--out2', type=str, default='grafik_pengujian2.png',
                        help='Nama output Pengujian 2 (default: grafik_pengujian2.png)')
    args = parser.parse_args()

    df = load_hasil(args.csv)
    cetak_tabel_ringkasan(df)
    buat_pengujian1(df, args.out1)
    buat_pengujian2(df, args.out2)
    print("\n[SELESAI] Kedua grafik berhasil dibuat.")


if __name__ == '__main__':
    main()