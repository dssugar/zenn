---
title: "GMKTec K8 + OCuLink eGPU (RTX 5060 Ti) を Proxmox で動かすまで"
emoji: "🖥️"
type: "tech"
topics: ["proxmox", "nvidia", "gpu", "oculink", "homelab"]
published: true
---

## はじめに

GMKTec K8（AMD Ryzen 8845HS）に OCuLink 経由で RTX 5060 Ti を接続し、Proxmox VE 上で認識させるまでの記録です。

シンプルに見えて、**5つの落とし穴**がありました。同じ構成で試す方の参考になれば。

### 環境

| 項目 | スペック |
|------|---------|
| Mini PC | GMKTec K8 (AMD Ryzen 7 8845HS) |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| 接続 | OCuLink (PCIe 4.0 x4) |
| ホスト OS | Proxmox VE 8.x (Debian 12 Bookworm) |
| 用途 | ヘッドレス（AI 推論用、ディスプレイ出力不要） |

## 落とし穴 1: OCuLink を挿すと LAN が死ぬ

OCuLink ケーブルを接続して起動すると、**2.5GbE LAN (Intel I226-V) が使えなくなりました**。`ethtool` で確認すると `Link detected: no`。

### 原因

OCuLink 接続により PCIe バス番号が再配置され、NIC のデバイス名が変わります（例: `enp1s0` → `enp2s0`）。Proxmox の `/etc/network/interfaces` は旧デバイス名を参照しているため、ネットワークが起動しません。

### 解決策: MAC アドレスベースの安定デバイス名

PCIe バス番号に依存しない命名にすることで、OCuLink の有無に関わらず同じデバイス名を維持します。

```bash
# /etc/systemd/network/10-lan-stable.link
[Match]
MACAddress=xx:xx:xx:xx:xx:xx  # 使用する NIC の MAC アドレス

[Link]
Name=lan0
```

```bash
# /etc/network/interfaces
iface lan0 inet manual

auto vmbr0
iface vmbr0 inet static
    address 192.168.1.161/24
    gateway 192.168.1.1
    bridge-ports lan0
    bridge-stp off
    bridge-fd 0
```

さらに、GRUB のカーネルパラメータに `pci=realloc` を追加します。

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet pci=realloc"
```

```bash
update-grub
```

:::message alert
`pci=realloc,assign-busses` は **カーネル 6.14 で GPU が認識されなくなる**ため、`assign-busses` は付けないでください。
:::

## 落とし穴 2: eGPU の起動順序が超重要

何度再起動しても GPU が `lspci` に表示されず、原因の切り分けに時間がかかりました。

### 正しい起動順序

```
1. eGPU ボックスの電源を ON（ファンが回ることを確認）
2. Proxmox を起動
```

**逆だと GPU は認識されません。** Thunderbolt/USB4 PCIe トンネルは、起動時にリンクアップしないとデバイスを検出できないようです。

:::message alert
**OCuLink はホットプラグ非対応です。** 稼働中にケーブルを抜き差しするとシステムがフリーズします。必ず電源を落としてから接続してください。
:::

## 落とし穴 3: カーネル 6.8 では open kernel module が動かない

RTX 5060 Ti（Blackwell 世代）の NVIDIA ドライバには **open kernel module** が必要です。

```
WARNING: The 'MIT/GPL' kernel modules are incompatible with the GPU(s) detected on this system.
```

Proxmox のデフォルトカーネル 6.8 では open kernel module のビルドに必要な機能が不足しており、このエラーが出ます。

### 解決策: カーネル 6.14 にアップグレード

```bash
apt-get install -y proxmox-kernel-6.14.11-6-bpo12-pve \
                   pve-headers-6.14.11-6-bpo12-pve
```

再起動後、`uname -r` で `6.14.x` になっていることを確認してください。

## 落とし穴 4: ドライバは 570 系が正解（580 ではない）

直感的には最新の 580 を入れたくなりますが、**RTX 5060 Ti は 570 系ドライバでサポートされています**。

580 を入れると以下のエラーが出ます：

```
WARNING: You do not appear to have an NVIDIA GPU supported by the 580.76.05
NVIDIA Linux graphics driver installed in this system.
```

### 正しいインストール手順

```bash
# ドライバをダウンロード
wget https://download.nvidia.com/XFree86/Linux-x86_64/570.211.01/NVIDIA-Linux-x86_64-570.211.01.run

# nouveau をブラックリスト
cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u

# 再起動後にインストール（open kernel module を指定）
chmod +x NVIDIA-Linux-x86_64-570.211.01.run
./NVIDIA-Linux-x86_64-570.211.01.run --silent --dkms --kernel-module-type=open
```

:::message
NVIDIA のサポートチップリストで確認できます:
https://download.nvidia.com/XFree86/Linux-x86_64/570.211.01/README/supportedchips.html
RTX 5060 Ti (Device ID: 2D04) は 570.211.01 でサポートされています。
:::

## 落とし穴 5: BIOS 設定

以下の設定が必要です（GMKTec K8 の BIOS はフルアンロック済み）。

| 設定項目 | 値 | 場所 |
|---------|-----|------|
| Above 4G Decoding | **Enabled** | PCI Subsystem Settings |
| Re-Size BAR Support | **Enabled** | PCI Subsystem Settings |
| SR-IOV Support | **Enabled** | PCI Subsystem Settings |
| IOMMU | **Enabled** | Advanced > AMD CBS > NBIO Common Options |

**Primary Display は iGPU のまま**にしてください。eGPU に切り替えると、ヘッドレス運用時にコンソールが見えなくなります。

## 最終結果

```
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 5060 Ti     Off |   00000000:01:00.0 Off |                  N/A |
| 30%   30C    P0             21W /  180W |       0MiB /  16311MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

## まとめ

| ハマりポイント | 解決策 |
|--------------|--------|
| OCuLink 接続で LAN が死ぬ | MAC アドレスベースの NIC 命名 + `pci=realloc` |
| GPU が認識されない | eGPU 電源 ON → Proxmox 起動の順序を守る |
| open kernel module が動かない | カーネル 6.8 → 6.14 にアップグレード |
| ドライバ 580 で非対応エラー | 570.211.01 が正解 |
| BIOS 設定 | Above 4G Decoding, IOMMU 等を有効化 |

**最大の教訓: OCuLink のホットプラグは厳禁。** 稼働中に触るとシステムがフリーズします。
